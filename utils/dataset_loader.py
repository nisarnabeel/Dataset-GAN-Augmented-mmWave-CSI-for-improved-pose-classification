import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def dataset_loader(data_path="data", batch_size=32, test_size=0.25):
    """
    Load CSI dataset for training WGAN.

    Args:
        data_path (str): Directory containing data_mm.pth and labels.pth
        batch_size (int): DataLoader batch size
        test_size (float): Test split ratio

    Returns:
        train_loader, test_loader, num_classes, input_shape
    """
    print(f"Loading CSI dataset from: {data_path}")

    data_file = os.path.join(data_path, "data_mm.pth")
    labels_file = os.path.join(data_path, "labels.pth")

    # Load raw data
    tdata = torch.load(data_file)
    tlabels = torch.load(labels_file)

    print(f"  -> Data: {data_file}")
    print(f"  -> Labels: {labels_file}")

    # Reshape to [N, 1, 30, 50]
    tdata = torch.transpose(tdata, 2, 1).unsqueeze(dim=1).float()

    # Normalize to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    tdata = torch.tensor(
        scaler.fit_transform(tdata.reshape(-1, tdata.shape[-1])).reshape(tdata.shape),
        dtype=torch.float32
    )

    tlabels = tlabels.to(torch.int64)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        tdata, tlabels, test_size=test_size, random_state=42
    )

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    num_classes = len(torch.unique(tlabels))
    input_shape = (1, tdata.shape[2], tdata.shape[3])

    print("Dataset loaded successfully!")
    print(f" - Training samples: {len(train_dataset)}")
    print(f" - Test samples: {len(test_dataset)}")
    print(f" - Number of classes: {num_classes}")
    print(f" - Input shape: {input_shape}")

    return train_loader, test_loader, num_classes, input_shape
