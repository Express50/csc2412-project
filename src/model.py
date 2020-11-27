from torch import Tensor
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from transformers import AutoConfig, AutoModel


import inspect

def get_args(func):
    sig = inspect.signature(func)
    kwargs = []

    for param in sig.parameters.values():
        if param.kind != param.VAR_KEYWORD:
            kwargs.append(param.name)

    return kwargs

# TODO: may need to freeze all but top layers of transformer for lower mem
class SentimentAnalysisModel(Module):
    def __init__(self, model_name: str, output_dim: int) -> None:
        super(SentimentAnalysisModel, self).__init__()

        config = AutoConfig.from_pretrained(model_name)

        self.transformer = AutoModel.from_pretrained(model_name)
        self.pre_classifier = Linear(config.hidden_size, config.hidden_size)
        self.dropout = Dropout(0.3)
        self.classifier = Linear(config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids) -> Tensor:
        # Some models don't require token_type_ids, so only pass supported args to forward()
        all_args = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

        supported_args = get_args(self.transformer.forward)
        args = {}
        for arg in supported_args:
            if arg in all_args:
                args[arg] = all_args[arg]


        transformer_output = self.transformer(**args)

        hidden_state = transformer_output[0]

        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output


# if __name__ == "__main__":
#     SentimentAnalysisModel('roberta-base', 5)
