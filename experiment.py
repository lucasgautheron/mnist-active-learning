# pylint: disable=unused-import,abstract-method

import logging

import psynet.experiment

from psynet.asset import asset, CachedAsset  # noqa
from psynet.bot import Bot
from psynet.modular_page import (
    ModularPage,
    ImagePrompt,
    DropdownControl,
)
from psynet.experiment import Participant
from psynet.timeline import Timeline, ModuleState

from psynet.trial.main import Trial
from psynet.trial.static import StaticNode, StaticTrial, StaticTrialMaker

from dallinger import db
from sqlalchemy import Column, LargeBinary, Integer
from sqlalchemy.ext.declarative import declared_attr

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

import csv

logging.basicConfig(
    level=logging.DEBUG,
)
logger = logging.getLogger()

SETUP = "adaptive"


class BayesianClassifier(nn.Module):
    """Bayesian Neural Network for MNIST classification using Monte Carlo Dropout"""

    def __init__(self, input_size, num_classes=10, dropout_rate=0.3):
        super(BayesianClassifier, self).__init__()
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.layers(x)

    def predict_with_uncertainty(self, x, n_samples=50):
        """Perform Monte Carlo sampling to estimate uncertainty"""
        self.train()  # Keep dropout active
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                pred = torch.softmax(logits, dim=1)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)
        return predictions


class ActiveLearning:
    def __init__(self, embedding_dim=512, num_classes=10):
        self.model = BayesianClassifier(embedding_dim, num_classes)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.model.to(self.device)

        self.dataset = pd.read_parquet(
            "static/mnist_complete_embeddings.parquet",
        ).set_index("dataset_index")
        self.embeddings = self.dataset[
            "clip_embedding"].apply(
            lambda x: np.array(x, dtype=np.float32),
        ).to_dict()
        self.labels = self.dataset["label"].to_dict()

    def _prepare_training_data(self, data):
        """Extract and prepare training data from the data structure"""
        X_train = []
        y_train = []

        for node_id, node_data in data["nodes"].items():
            if "x" in node_data:
                embedding = self.embeddings[node_data["x"]]

                # Collect all labels for this node from different trials
                labels = []
                for trial_id, trial_data in node_data["trials"].items():
                    if trial_data["y"] is not None:
                        labels.append(trial_data["y"])

                # Use majority vote or most recent label
                if labels:
                    # Use most common label (majority vote)
                    from collections import Counter
                    most_common_label = Counter(labels).most_common(1)[0][0]
                    X_train.append(embedding)
                    y_train.append(most_common_label)

        if len(X_train) == 0:
            return None, None

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        return X_train, y_train

    def _train_model(self, X_train, y_train, epochs=100):
        """Train the Bayesian model on available data"""
        if len(X_train) < 2:  # Need at least 2 samples
            logger.warning("Not enough training data for model training")
            return False

        # Normalize features
        X_scaled = self.scaler.fit_transform(X_train)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()

        # Simple batch training (could be improved with DataLoader for larger datasets)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logger.debug(f"Training epoch {epoch}, loss: {loss.item():.4f}")

        self.is_trained = True
        logger.info(f"Model trained on {len(X_train)} samples")
        return True

    def _compute_bald_scores(self, candidates_data, n_samples=100):
        """Compute BALD scores for candidate nodes"""
        if not self.is_trained:
            logger.warning("Model not trained, returning random scores")
            return np.random.rand(len(candidates_data))

        # Prepare candidate embeddings
        candidate_embeddings = []
        for node_id in candidates_data:
            if "x" in candidates_data[node_id]:
                embedding = self.embeddings[candidates_data[node_id]["x"]]
                candidate_embeddings.append(embedding)

        if len(candidate_embeddings) == 0:
            return np.array([])

        candidate_embeddings = np.array(candidate_embeddings)

        # Normalize using the same scaler
        X_scaled = self.scaler.transform(candidate_embeddings)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Get Monte Carlo predictions
        mc_predictions = self.model.predict_with_uncertainty(
            X_tensor, n_samples,
        )

        # Compute BALD scores
        # BALD = H(y|x) - E[H(y|θ,x)]

        # Predictive entropy: H(y|x)
        mean_predictions = np.mean(mc_predictions, axis=0)
        predictive_entropy = -np.sum(
            mean_predictions * np.log(mean_predictions + 1e-10), axis=1,
        )

        # Expected entropy: E[H(y|θ,x)]
        individual_entropies = -np.sum(
            mc_predictions * np.log(mc_predictions + 1e-10), axis=2,
        )
        expected_entropy = np.mean(individual_entropies, axis=0)

        # BALD score = Mutual Information
        bald_scores = predictive_entropy - expected_entropy

        logger.info(
            f"BALD scores computed for {len(candidates_data)} candidates",
        )
        logger.info(
            f"BALD score range: [{bald_scores.min():.4f}, {bald_scores.max():.4f}]",
        )

        return bald_scores

    def get_optimal_node(self, candidates, participant, data):
        """
        Select the optimal node using BALD active learning

        Args:
            candidates: List of candidate nodes
            participant: Current participant
            data: Dictionary containing nodes and participants data

        Returns:
            Selected node ID with highest BALD score
        """
        logger.info(
            f"Active learning selection for {len(candidates)} candidates",
        )

        # Prepare training data from existing trials
        X_train, y_train = self._prepare_training_data(data)

        if X_train is not None and len(X_train) > 1:
            # Train/retrain model on available data
            self._train_model(X_train, y_train)

            # Prepare candidate data
            candidates_data = {}
            for candidate in candidates:
                if candidate in data["nodes"]:
                    candidates_data[candidate] = data["nodes"][candidate]

            # Compute BALD scores
            bald_scores = self._compute_bald_scores(candidates_data)

            self.model.eval()
            test_nodes = set(self.dataset.index.values)
            X_test = [self.embeddings[node] for node in test_nodes]
            y_test = [self.labels[node] for node in test_nodes]

            X_scaled = self.scaler.fit_transform(X_test)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            accuracy = np.mean(predictions == y_test)
            logger.info(accuracy)

            with open(f"output/utility_{SETUP}.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        accuracy,
                    ],
                )

            # from matplotlib import pyplot as plt
            # import seaborn as sns
            # from sklearn.metrics import confusion_matrix
            # confusion = confusion_matrix(y_test, predictions, normalize="true")
            # sns.heatmap(confusion, cmap="Reds")
            # plt.show()

            if len(bald_scores) > 0:
                # Select candidate with highest BALD score
                best_idx = np.argmax(bald_scores)
                selected_node_id = list(candidates_data.keys())[best_idx]

                logger.info(
                    f"Selected node {selected_node_id} with BALD score {bald_scores[best_idx]:.4f}",
                )

                # Find and return the actual candidate node
                return selected_node_id

        # Fallback: return first candidate if active learning fails
        logger.info("Active learning failed, using fallback selection")
        return candidates[0] if candidates else None


class ImageNode(StaticNode):
    # @declared_attr
    # def embedding(cls):
    #     return cls.__table__.c.get("embedding", Column(LargeBinary))

    def __init__(self, *args, embedding, **kwargs):
        super().__init__(*args, **kwargs)
        # self.embedding = np.array(embedding, dtype=np.float32).tobytes()


images = pd.read_parquet("static/mnist_complete_embeddings.parquet")

nodes = [
    ImageNode(
        definition={
            "idx": image["dataset_index"],
            "image_path": image["image_path"],
            "dimensions": len(image["clip_embedding"]),
            "label": int(image["label"]),
        },
        assets={
            "stimulus": asset(
                f"static/{image['image_path']}",
            ),
        },
        embedding=image["clip_embedding"],
    )
    for image in images.to_dict(orient="records")
]


class ImageTrialMaker(StaticTrialMaker):
    def __init__(self, *args, optimizer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer() if optimizer is not None else None

    def prior_data(self):
        data = {"nodes": dict(), "participants": dict()}

        # List participants involved in this trial maker
        participants = (
            db.session.query(Participant)
            .join(Participant._module_states)
            .filter(
                ModuleState.module_id == self.id, ModuleState.started == True,
            )
            .distinct()
            .all()
        )

        data["participants"] = {
            participant.id: dict()
            for participant in participants
        }

        # Fetch all nodes related to this trial maker
        networks = self.network_class.query.filter_by(
            trial_maker_id=self.id, full=False, failed=False,
        ).all()
        nodes = [network.head for network in networks]

        # Fetch all trials that belong to this trial maker
        trials = Trial.query.filter(
            Trial.failed == False,
            Trial.finalized == True,
            Trial.is_repeat_trial == False,
            Trial.trial_maker_id == self.id,
        ).all()

        trials_by_node = {}
        for trial in trials:
            if trial.node_id not in trials_by_node:
                trials_by_node[trial.node_id] = []
            trials_by_node[trial.node_id].append(trial)

        # Process trials for each node
        for node in nodes:
            data["nodes"][node.id] = {
                "x": node.definition["idx"],
                "trials": dict(),
            }

            if node.id in trials_by_node:
                data["nodes"][node.id]["trials"] = {
                    trial.id: {
                        "y": trial.y,
                        "participant_id": trial.participant_id,
                    }
                    for trial in trials_by_node[node.id]
                    if trial.y is not None
                }

        return data

    def prioritize_networks(self, networks, participant, experiment):
        if self.optimizer is None:
            return networks

        node_network = {
            network.head.id: i for i, network in enumerate(networks)
        }

        # retrieve all relevant prior data
        data = self.prior_data()

        # retrieve optimal node
        next_node = self.optimizer.get_optimal_node(
            list(node_network.keys()), participant, data,
        )

        return [networks[node_network[next_node]]]

    def finalize_trial(
            self,
            answer,
            trial,
            experiment,
            participant,
    ):
        trial.y = int(answer)


class ImageTrial(StaticTrial):
    time_estimate = 5

    @declared_attr
    def y(cls):
        return cls.__table__.c.get("y", Column(Integer))

    def show_trial(self, experiment, participant):
        return ModularPage(
            "classification",
            ImagePrompt(
                self.assets["stimulus"],
                "What is the number on this image?",
                width=200,
                height=200,
            ),
            DropdownControl(
                choices=np.arange(10),
                labels=[f"{i}" for i in np.arange(10)],
            ),
            time_estimate=self.time_estimate,
            bot_response=lambda: self.definition["label"],
        )


class Exp(psynet.experiment.Experiment):
    label = "MNIST"

    timeline = Timeline(
        ImageTrialMaker(
            id_="image_classification",
            trial_class=ImageTrial,
            nodes=nodes,
            max_trials_per_participant=100,
            expected_trials_per_participant=100,
            target_n_participants=1,
            recruit_mode="n_participants",
            optimizer=ActiveLearning if SETUP == "adaptive" else None,

        ),
    )

    def test_check_bot(self, bot: Bot, **kwargs):
        assert len(bot.alive_trials) == len(nodes)
