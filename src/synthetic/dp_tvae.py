import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import pandas as pd
import numpy as np
from tqdm import tqdm

from ctgan.synthesizers.tvae import TVAE, Encoder, Decoder, _loss_function, random_state
from ctgan.data_transformer import DataTransformer
from sdv.single_table import TVAESynthesizer
from sdv.single_table.ctgan import detect_discrete_columns
from opacus import PrivacyEngine

class DPDecoder(nn.Module):
    """
    Custom Decoder for TVAE that returns a single concatenated tensor of shape
    (batch_size, data_dim * 2) instead of a tuple, to avoid Opacus functorch
    per-sample gradient failures on tuple outputs.
    """
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(DPDecoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [nn.Linear(dim, item), nn.ReLU()]
            dim = item

        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)
        self.sigma = nn.Parameter(torch.ones(data_dim) * 0.1)
        self.data_dim = data_dim

    def forward(self, input_):
        recon = self.seq(input_)
        sigmas = self.sigma.unsqueeze(0).repeat(input_.size(0), 1)
        return torch.cat([recon, sigmas], dim=1)

class JointModule(nn.Module):
    """
    Joint container model for TVAE Encoder and DPDecoder, allowing Opacus
    to compute per-sample gradients cleanly on a single forward/backward pass.
    """
    def __init__(self, encoder, decoder):
        super(JointModule, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, std, logvar = self.encoder(x)
        eps = torch.randn_like(std)
        emb = eps * std + mu
        out = self.decoder(emb)
        
        # Split the concatenated output of DPDecoder
        rec = out[:, :self.decoder.data_dim]
        sigmas = out[0, self.decoder.data_dim:]
        return rec, sigmas, mu, logvar

class DPTVAE(TVAE):
    """
    Differentially Private TVAE model subclassing CTGAN's TVAE.
    """
    def __init__(
        self,
        target_epsilon=1.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=False,
        verbose=False
    ):
        super(DPTVAE, self).__init__(
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2scale=l2scale,
            batch_size=batch_size,
            epochs=epochs,
            loss_factor=loss_factor,
            cuda=cuda,
            verbose=verbose
        )
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm

    @random_state
    def fit(self, train_data, discrete_columns=()):
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        
        # Using drop_last=True for constant batch sizing with Opacus
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = DPDecoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale
        )

        m = JointModule(encoder, self.decoder).to(self._device)
        
        privacy_engine = PrivacyEngine()
        pm, po, pl = privacy_engine.make_private_with_epsilon(
            module=m,
            optimizer=optimizerAE,
            data_loader=loader,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm
        )

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = 'Loss: {loss:.3f}'
            iterator.set_description(iterator_description.format(loss=0))

        for i in iterator:
            loss_values = []
            batch = []
            for id_, data in enumerate(pl):
                po.zero_grad()
                real = data[0].to(self._device)
                rec, sigmas, mu, logvar = pm(real)
                
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                po.step()
                
                # Clamp sigma Parameter values to keep VAE decoding stable
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                batch.append(id_)
                loss_values.append(loss.detach().cpu().item())

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i] * len(batch),
                'Batch': batch,
                'Loss': loss_values
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self.verbose:
                iterator.set_description(
                    iterator_description.format(
                        loss=loss.detach().cpu().item()))

    @random_state
    def sample(self, samples):
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake_sigmas = self.decoder(noise)
            fake = fake_sigmas[:, :self.decoder.data_dim]
            sigmas = fake_sigmas[0, self.decoder.data_dim:]
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

class DPTVAESynthesizer(TVAESynthesizer):
    """
    Differentially Private TVAE Synthesizer for SDV.
    """
    def __init__(self, target_epsilon=1.0, target_delta=1e-5, max_grad_norm=1.0, *args, **kwargs):
        super(DPTVAESynthesizer, self).__init__(*args, **kwargs)
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        
        self._model_kwargs['target_epsilon'] = target_epsilon
        self._model_kwargs['target_delta'] = target_delta
        self._model_kwargs['max_grad_norm'] = max_grad_norm

    def _fit(self, processed_data):
        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.get_metadata(),
            processed_data,
            transformers
        )
        self._model = DPTVAE(**self._model_kwargs)
        self._model.fit(processed_data, discrete_columns=discrete_columns)
