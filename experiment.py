import psynet.experiment
from psynet.asset import asset  # noqa
from psynet.bot import Bot
from psynet.modular_page import (
    ModularPage,
    ImagePrompt,
    DropdownControl,
)
from psynet.timeline import Event, ProgressDisplay, ProgressStage, Timeline
from psynet.trial.static import StaticNode, StaticTrial, StaticTrialMaker

import numpy as np
import pandas as pd

nodes = [
    StaticNode(
        definition={
            "idx": frequency_gradient,
            "start_frequency": start_frequency,
            "frequencies": [start_frequency + i * frequency_gradient for i in
                            range(5)],
        },
        assets={
            "stimulus": asset(
                synth_stimulus,
                extension=".wav",
                on_demand=True,
            ),
        },
    )
    for frequency_gradient in [-100, 0, 100]
    for start_frequency in [-100, 0, 100]
]


class ImageTrialMaker(StaticTrialMaker):
    def prioritize_networks(self, networks, experiment):
        return networks


class ImageTrial(StaticTrial):
    time_estimate = 5

    def show_trial(self, experiment, participant):
        return ModularPage(
            "imitation",
            ImagePrompt(
                self.assets["stimulus"],
                "Please imitate the spoken word as closely as possible.",
            ),
            DropdownControl(
                choices=np.arange(10),
                labels=[f"{i}" for i in np.arange(10)],
            ),
            time_estimate=self.time_estimate,
        )


class Exp(psynet.experiment.Experiment):
    label = "Static audio demo"

    timeline = Timeline(
        StaticTrialMaker(
            id_="image_classification",
            trial_class=ImageTrial,
            nodes=nodes,
            expected_trials_per_participant=100,
            target_n_participants=1,
            recruit_mode="n_participants",
        ),
    )

    def test_check_bot(self, bot: Bot, **kwargs):
        assert len(bot.alive_trials) == len(nodes)
