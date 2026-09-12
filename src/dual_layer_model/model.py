import torch
import torch.nn as nn


class ConvFeatureAttentionLSTM(nn.Module):
    """
    Brain-inspired temporal classifier:
    local temporal filtering -> recurrent state memory -> temporal attention -> class logits.

    The forward method can return internal states so the model can be compared with
    physiological 10 ms neural dynamics and used for mechanistic ablation analysis.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        attention_hidden_size: int = 128,
        num_classes: int = 5,
        conv_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.35,
        time_attn_activation: str = "tanh",
        ablate_conv: bool = False,
        ablate_lstm: bool = False,
        ablate_attention: bool = False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.attention_hidden_size = int(attention_hidden_size)
        self.num_classes = int(num_classes)
        self.conv_channels = int(conv_channels)
        self.ablate_conv = bool(ablate_conv)
        self.ablate_lstm = bool(ablate_lstm)
        self.ablate_attention = bool(ablate_attention)

        if self.ablate_conv:
            self.conv = nn.Identity()
            recurrent_input_size = self.input_size
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.input_size,
                    out_channels=self.conv_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode="replicate",
                ),
                nn.BatchNorm1d(self.conv_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            recurrent_input_size = self.conv_channels

        if self.ablate_lstm:
            self.input_projection = nn.Linear(recurrent_input_size, self.hidden_size)
            self.lstm = None
        else:
            self.input_projection = None
            self.lstm = nn.LSTM(
                input_size=recurrent_input_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        if time_attn_activation.lower() == "relu":
            attn_act = nn.ReLU()
        elif time_attn_activation.lower() == "tanh":
            attn_act = nn.Tanh()
        else:
            attn_act = nn.Identity()

        self.time_attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.attention_hidden_size),
            attn_act,
            nn.Linear(self.attention_hidden_size, 1),
        )
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name:
                if param.dim() >= 2:
                    if "conv" in name:
                        nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
                    else:
                        nn.init.xavier_normal_(param)
                else:
                    nn.init.ones_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor, return_states: bool = False):
        """
        Parameters
        ----------
        x:
            Tensor shaped (batch, features, seq_len).
        return_states:
            If True, return logits and internal representations.
        """
        conv_out = self.conv(x)
        seq_features = conv_out.permute(0, 2, 1)

        if self.ablate_lstm:
            lstm_out = torch.tanh(self.input_projection(seq_features))
        else:
            lstm_out, _ = self.lstm(seq_features)

        if self.ablate_attention:
            time_attn = torch.full(
                (x.shape[0], lstm_out.shape[1]),
                fill_value=1.0 / lstm_out.shape[1],
                dtype=lstm_out.dtype,
                device=lstm_out.device,
            )
        else:
            time_attn = self.time_attention(lstm_out).squeeze(-1)
            time_attn = torch.softmax(time_attn, dim=1)

        context = torch.bmm(time_attn.unsqueeze(1), lstm_out).squeeze(1)
        logits = self.fc(context)

        if not return_states:
            return logits
        return {
            "logits": logits,
            "conv_out": conv_out,
            "lstm_out": lstm_out,
            "time_attention": time_attn,
            "context": context,
        }
