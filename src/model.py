from torch import Tensor
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from transformers import AutoConfig, AutoModel


class SentimentAnalysisModel(Module):
    def __init__(self, model_name: str, output_dim: int) -> None:
        super(SentimentAnalysisModel, self).__init__()

        config = AutoConfig.from_pretrained(model_name)

        self.transformer = AutoModel.from_pretrained(model_name)
        self.pre_classifier = Linear(config.hidden_size, config.hidden_size)
        self.dropout = Dropout(0.3)
        self.classifier = Linear(config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids) -> Tensor:
        transformer_output = self.transformer(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              token_type_ids=token_type_ids)
        hidden_state = transformer_output[0]

        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output


if __name__ == "__main__":
    SentimentAnalysisModel('roberta-base', 5)
