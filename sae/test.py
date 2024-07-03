import torch
import unittest
from model import SparseAutoencoder, SAEConfig
from sae.trainer import train_autoencoder

class TestSparseAutoencoder(unittest.TestCase):
    def setUp(self):
        self.input_dim = 4
        self.hidden_dim = 24
        self.real_features = 32
        self.sparsity = 0.2
        self.no_active = int(self.real_features * self.sparsity)
        self.iterations = 2**12  # Reduced for quicker testing
        self.directions = torch.randn(self.real_features, self.input_dim)
        
        self.data = self.generate_test_data()
        
        self.config = SAEConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            layer_num=1,
            l1_penalty=2,
            num_epochs=100,  # Reduced for quicker testing
            batch_size=32,
            learning_rate=1e-3
        )
        
        self.model = SparseAutoencoder(self.config)

    def generate_test_data(self):
        data = []
        for _ in range(self.iterations):
            indices = torch.randperm(self.real_features)[:self.no_active]
            acts = (torch.randint(-2, 2, (self.no_active,1)) * self.directions[indices]).sum(dim=0)
            data.append(acts)
        return torch.stack(data)

    def test_model_initialization(self):
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.encoder.weight.shape, (self.hidden_dim, self.input_dim))
        self.assertEqual(self.model.decoder.weight.shape, (self.input_dim, self.hidden_dim))

    def test_forward_pass(self):
        sample = self.data[0].unsqueeze(0)
        decoded, encoded = self.model(sample)
        self.assertEqual(decoded.shape, sample.shape)
        self.assertEqual(encoded.shape, (1, self.hidden_dim))

    def test_training(self):
        train_autoencoder(self.model, self.data, self.config)


if __name__ == '__main__':
    unittest.main()