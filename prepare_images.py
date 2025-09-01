import os
import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision import datasets, transforms
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from tqdm import tqdm


def export_mnist_with_clip_embeddings(
        output_dir='static', dataset_type='train',
):
    """
    Export MNIST images to a folder and create a parquet file with CLIP embeddings.

    Args:
        output_dir (str): Directory to save images and parquet file
        dataset_type (str): 'train' or 'test' to specify which dataset split to export
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Load CLIP model and processor
    print("Loading CLIP model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load MNIST dataset
    print(f"Loading MNIST {dataset_type} dataset...")
    transform = transforms.Compose([transforms.ToTensor()])

    is_train = (dataset_type == 'train')
    mnist_dataset = datasets.MNIST(
        root='./data',
        train=is_train,
        download=True,
        transform=transform,
    )

    # Prepare data for parquet
    data_records = []

    N = len(mnist_dataset)
    N = 1024

    print(f"Processing {N} images with CLIP embeddings...")

    # Process images in batches for efficiency
    batch_size = 32
    for batch_start in tqdm(
            range(0, N, batch_size), desc="Processing batches",
    ):
        batch_end = min(batch_start + batch_size, N)
        batch_images = []
        batch_metadata = []

        # Prepare batch
        for idx in range(batch_start, batch_end):
            image_tensor, label = mnist_dataset[idx]

            # Convert tensor to PIL Image (grayscale to RGB for CLIP)
            image_array = (image_tensor.squeeze().numpy() * 255).astype('uint8')
            pil_image = Image.fromarray(image_array, mode='L')
            pil_image_rgb = pil_image.convert('RGB')  # CLIP expects RGB

            # Create filename and save image
            filename = f"{dataset_type}_{idx:06d}_label_{label}.png"
            image_path = os.path.join(images_dir, filename)
            pil_image.save(image_path)  # Save as grayscale

            batch_images.append(pil_image_rgb)
            batch_metadata.append(
                {
                    'dataset_index': idx,
                    'filename': filename,
                    'image_path': os.path.join('images', filename),
                    'label': label,
                    'dataset_type': dataset_type,
                },
            )

        # Process batch through CLIP
        with torch.no_grad():
            inputs = processor(
                images=batch_images, return_tensors="pt", padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get image features (CLS embeddings)
            image_features = model.get_image_features(**inputs)

            # Normalize embeddings (as is standard practice with CLIP)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True,
            )

            # Move back to CPU and convert to numpy
            embeddings = image_features.cpu().numpy()

        # Add embeddings to metadata
        for i, metadata in enumerate(batch_metadata):
            # Convert numpy array to list for parquet compatibility
            metadata['clip_embedding'] = embeddings[i].tolist()
            data_records.append(metadata)

    # Create DataFrame
    df = pd.DataFrame(data_records)

    # Save to parquet
    parquet_path = os.path.join(
        output_dir, f'mnist_{dataset_type}_embeddings.parquet',
    )
    df.to_parquet(parquet_path, index=False)

    print(f"\nExport completed!")
    print(f"Images saved to: {images_dir}")
    print(f"Parquet file saved to: {parquet_path}")
    print(f"Total images exported: {len(data_records)}")
    print(f"CLIP embedding dimension: {embeddings.shape[1]}")

    # Print summary statistics
    print(f"\nDataset summary:")
    print(f"Labels distribution:")
    print(df['label'].value_counts().sort_index())

    return df


def export_both_datasets(output_dir='static'):
    """
    Export both training and test datasets with CLIP embeddings.
    """
    print("Exporting MNIST Training Dataset with CLIP embeddings...")
    train_df = export_mnist_with_clip_embeddings(output_dir, 'train')

    print("\nExporting MNIST Test Dataset with CLIP embeddings...")
    test_df = export_mnist_with_clip_embeddings(output_dir, 'test')

    # Create combined parquet file
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_parquet_path = os.path.join(
        output_dir, 'mnist_complete_embeddings.parquet',
    )
    combined_df.to_parquet(combined_parquet_path, index=False)

    print(f"\nCombined parquet file saved to: {combined_parquet_path}")
    print(f"Total images in combined dataset: {len(combined_df)}")

    return train_df, test_df, combined_df


if __name__ == "__main__":
    train_df, test_df, combined_df = export_both_datasets('static')
