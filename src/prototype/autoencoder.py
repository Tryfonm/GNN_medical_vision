import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        conv_out_channels: int = 8,
        embedding_size: int = 9,
        dim_0: int = 32,
        dim_1: int = 32,
        dim_2: int = 28,
    ):
        super(Encoder, self).__init__()
        self.conv_out_channels = conv_out_channels
        self.embedding_size = embedding_size
        self._dim_0 = dim_0 // 2
        self._dim_1 = dim_1 // 2
        self._dim_2 = dim_2 // 2

        self.conv1 = nn.Conv3d(1, self.conv_out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool3d(2, padding=0, stride=2)
        self.fc = nn.Linear(
            self.conv_out_channels * self._dim_0 * self._dim_1 * self._dim_2,
            self.embedding_size,
        )

    def forward(self, x):
        x = self.conv1(x)  # torch.Size([batch, channels, dim0, dim1, dim2])
        x = self.relu(x)
        x = self.pool(x)  # torch.Size([batch, channels, dim0/2, dim1/2, dim2/2])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, conv_in_channels, embedding_size=9, dim_0=32, dim_1=32, dim_2=28
    ):
        super(Decoder, self).__init__()
        self.conv_in_channels = conv_in_channels
        self.embedding_size = embedding_size
        self._dim_0 = dim_0 // 2
        self._dim_1 = dim_1 // 2
        self._dim_2 = dim_2 // 2

        self.fc = nn.Linear(
            self.embedding_size,
            self.conv_in_channels * self._dim_0 * self._dim_1 * self._dim_2,
        )
        self.conv_transpose = nn.ConvTranspose3d(
            conv_in_channels, 1, kernel_size=3, stride=1, padding=1
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(
            x.size(0), self.conv_in_channels, self._dim_0, self._dim_1, self._dim_2
        )
        x = self.upsample(x)
        x = self.conv_transpose(x)
        x = self.sigmoid(x)

        return x


class Autoencoder(nn.Module):
    def __init__(
        self,
        conv_in_channels,
        conv_out_channels,
        embedding_size=9,
        dim_0=32,
        dim_1=32,
        dim_2=28,
    ):
        super(Autoencoder, self).__init__()
        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.embedding_size = embedding_size
        self._dim_0 = dim_0
        self._dim_1 = dim_1
        self._dim_2 = dim_2

        self.encoder = Encoder(
            conv_out_channels=self.conv_out_channels,
            embedding_size=self.embedding_size,
            dim_0=self._dim_0,
            dim_1=self._dim_1,
            dim_2=self._dim_2,
        )
        self.decoder = Decoder(
            conv_in_channels=self.conv_in_channels,
            embedding_size=self.embedding_size,
            dim_0=self._dim_0,
            dim_1=self._dim_1,
            dim_2=self._dim_2,
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return encoder_output, decoder_output


if __name__ == "__main__":
    batch_size = 1
    conv_channels = 1
    dim_0 = 240
    dim_1 = 240
    dim_2 = 154
    x = torch.randn((batch_size, conv_channels, dim_0, dim_1, dim_2))

    autoencoder = Autoencoder(
        conv_in_channels=8,
        conv_out_channels=8,
        embedding_size=32,
        dim_0=240,
        dim_1=240,
        dim_2=154,
    )
    encoder_output, decoder_output = autoencoder(x)

    assert encoder_output.shape == torch.Size([batch_size, 32])
    assert decoder_output.shape == torch.Size(
        [batch_size, conv_channels, dim_0, dim_1, dim_2]
    )
