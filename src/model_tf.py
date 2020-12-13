# from torch import Tensor
# from torch.nn.modules.activation import ReLU
# from torch.nn.modules.dropout import Dropout
# from torch.nn.modules.linear import Linear
# from torch.nn.modules.module import Module
from tensorflow import Tensor
from tensorflow.keras.activations import relu as ReLU, sigmoid
from tensorflow.keras.layers import Dropout, Dense as Linear
from tensorflow.keras import Model
from transformers import AutoConfig, TFAutoModel

import inspect
from transformers.modeling_tf_distilbert import TFDistilBertModel

from transformers.modeling_tf_gpt2 import TFGPT2Model
from transformers.modeling_tf_t5 import TFT5Model

def get_args(func):
    sig = inspect.signature(func)
    kwargs = []

    for param in sig.parameters.values():
        if param.kind != param.VAR_KEYWORD:
            kwargs.append(param.name)

    return kwargs

# TODO: may need to freeze all but top layers of transformer for lower mem
class SentimentAnalysisModel(Model):
    def __init__(self, model_name: str, output_dim: int) -> None:
        super(SentimentAnalysisModel, self).__init__()

        config = AutoConfig.from_pretrained(model_name)

        self.transformer = TFAutoModel.from_pretrained(model_name)

        # freeze all but last layer of transformer
        layers_to_freeze = None
        frozen_params = 0
        if type(self.transformer) is TFGPT2Model:
            layers_to_freeze = self.transformer.layers[0].h[:-1]
        elif type(self.transformer) is TFDistilBertModel:
            layers_to_freeze = self.transformer.layers[0].transformer.layer[:-1]
        elif type(self.transformer) is TFT5Model:
            layers_to_freeze = self.transformer.layers[1].block[:-1]
            layers_to_freeze.extend(self.transformer.layers[2].block[:-1])

        for layer in layers_to_freeze:
            layer.trainable = False

        print(f'Init model: frozen {len(self.transformer.non_trainable_variables)} variables.')

        self.pre_classifier = Linear(units=config.hidden_size, input_dim=config.hidden_size, activation='linear')
        self.dropout = Dropout(0.3)
        # self.classifier = Linear(units=output_dim, input_dim=config.hidden_size, activation='linear')
        self.classifier = Linear(units=1, input_dim=config.hidden_size, activation='linear')

    def call(self, args): #input_ids, attention_mask, token_type_ids) -> Tensor:
        # Some models don't require token_type_ids, so only pass supported args to forward()
        input_ids = args.pop('input_ids')

        all_args = {
            'attention_mask': args["attention_mask"],
            'token_type_ids': args["token_type_ids"]
        }

        supported_args = get_args(self.transformer.call)
        args = {}
        for arg in supported_args:
            if arg in all_args:
                args[arg] = all_args[arg]

        transformer_output = self.transformer(input_ids, **args)

        hidden_state = transformer_output[0]

        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = ReLU(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output


if __name__ == "__main__":
    model = SentimentAnalysisModel('distilgpt2', 5)

    print(type(model.transformer))
