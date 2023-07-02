from typing import Optional

import torch
from allennlp.common import Registrable
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from torch.nn import LSTMCell, Module

import torch.nn.functional as F

class TranslationModel(Module, Registrable):

    def __init__(self, vocab: Vocabulary, maximum_fertility: int, target_namespace: str):
        super().__init__()
        self.vocab = vocab
        self.maximum_fertility = maximum_fertility
        self.target_namespace = target_namespace
        self.vocab_size = self.vocab.get_vocab_size(target_namespace)


    def get_input_dim(self) -> int:
        raise NotImplementedError()

    def forward(self, embedded_input: torch.Tensor, input_mask: torch.Tensor, fertilities: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        compute log probs for output
        :param input_mask: shape (batch_size, input_seq_len)
        :param embedded_input: shape (batch_size, input_seq_len, embedding_dim)
        :param fertilities: shape (batch_size, input_seq_len, max. fertility)
        :return: shape (batch_size, input_seq_len, maximum fertility + 1, vocab size)
        """
        raise NotImplementedError()


@TranslationModel.register("lexical_translation")
class LexicalTranslationModel(TranslationModel):

    def __init__(self, vocab: Vocabulary, maximum_fertility: int, target_namespace: str, mlp: FeedForward):
        super().__init__(vocab, maximum_fertility, target_namespace)

        self.maximum_fertility = maximum_fertility
        # self.positional_dim = positional_dim
        self.target_namespace = target_namespace

        # self.positional_embedding = torch.nn.Parameter(
        #     torch.randn([1, self.maximum_fertility + 1, self.positional_dim]),
        #     requires_grad=True)
        self.mlp = mlp
        self.output_layer = torch.nn.Linear(self.mlp.get_output_dim(), (self.maximum_fertility+1) * self.vocab_size)


    def get_input_dim(self) -> int:
        return self.mlp.get_input_dim()

    def forward(self, embedded_input: torch.Tensor, input_mask: torch.Tensor, fertilities = None) -> torch.Tensor:
        """
        compute log probs for output
        :param input_mask: shape (batch_size, input_seq_len)
        :param embedded_input: shape (batch_size, input_seq_len, embedding_dim)
        :return: shape (batch_size, input_seq_len, maximum fertility + 1, vocab size)
        """

        batch_size, input_seq_len, embedding_dim = embedded_input.shape

        # upscaled_positional_info = self.positional_embedding.unsqueeze(0).expand([batch_size, input_seq_len, self.maximum_fertility+1, self.positional_dim])

        # embedded_input = embedded_input.unsqueeze(2) #shape (batch_size, input_seq_len, 1, embedding_dim)

        # before_mlp = upscaled_positional_info + embedded_input #shape (batch_size, input_seq_len, self.maximum_fertility + 1, embedding dim)
        # before_mlp = torch.zeros_like(upscaled_positional_info) + embedded_input #shape (batch_size, input_seq_len, self.maximum_fertility + 1, embedding dim)

        # return torch.log_softmax(self.output_layer(self.mlp(before_mlp)), dim=-1)

        o = self.output_layer(self.mlp(embedded_input)) #shape (batch_size, input_seq_len, self.maximum fertility * vocab size)
        o = o.reshape([batch_size, input_seq_len, self.maximum_fertility+1, self.vocab.get_vocab_size(self.target_namespace)])
        # I hope the above doesn't break for specific dimensionalities...

        return torch.log_softmax(o, dim=-1)



@TranslationModel.register("lexical_translation2")
class LexicalTranslationModel2(TranslationModel):

    def __init__(self, vocab: Vocabulary, maximum_fertility: int, target_namespace: str, mlp: FeedForward):
        super().__init__(vocab, maximum_fertility, target_namespace)

        self.maximum_fertility = maximum_fertility
        # self.positional_dim = positional_dim
        self.target_namespace = target_namespace

        self.mlp = mlp

        self.fertility_embedding = torch.nn.Parameter(
            torch.randn([1, 1, self.maximum_fertility + 1, self.mlp.get_input_dim()]),
            requires_grad=True)

        self.output_layer = torch.nn.Linear(self.mlp.get_output_dim(), (self.maximum_fertility+1) * self.vocab_size)



    def get_input_dim(self) -> int:
        return self.mlp.get_input_dim()

    def forward(self, embedded_input: torch.Tensor, input_mask: torch.Tensor, fertilities : Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        compute log probs for output
        :param input_mask: shape (batch_size, input_seq_len)
        :param embedded_input: shape (batch_size, input_seq_len, embedding_dim)
        :return: shape (batch_size, input_seq_len, maximum fertility + 1, vocab size)
        """

        batch_size, input_seq_len, embedding_dim = embedded_input.shape

        weighted_embedding = self.fertility_embedding * fertilities.unsqueeze(3) #shape (batch_size, input_seq_len, max fertility, dim)

        o = self.output_layer(self.mlp(embedded_input + weighted_embedding.sum(dim=2))) #shape (batch_size, input_seq_len, self.maximum fertility * vocab size)
        o = o.reshape([batch_size, input_seq_len, self.maximum_fertility+1, self.vocab.get_vocab_size(self.target_namespace)])
        # I hope the above doesn't break for specific dimensionalities...

        return torch.log_softmax(o, dim=-1)




@TranslationModel.register("lstm_translation_model")
class LSTMTranslationModel(TranslationModel):

    def __init__(self, vocab: Vocabulary, maximum_fertility: int, target_namespace: str, input_dim: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.0,
                 soft_at_test: bool = False,
                 use_gumbel_noise: bool = False,
                 hard: bool = False):
        super().__init__(vocab, maximum_fertility, target_namespace)

        self.lstm = LSTMCell(input_dim + embedding_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout = dropout

        self.oov_id = self.vocab.get_token_index(self.vocab._oov_token, namespace=target_namespace)

        self.output_layer = torch.nn.Linear(hidden_dim, self.vocab.get_vocab_size(self.target_namespace))
        self.embedding_weight = torch.nn.Embedding(self.vocab.get_vocab_size(self.target_namespace), embedding_dim)

        self.range_tensor = None
        self.soft_at_test = soft_at_test
        self.use_gumbel_noise = use_gumbel_noise
        self.hard = hard

        if hard:
            assert not soft_at_test, "Doesn't make sense to use straight-through Gumbel at train, and soft at test!"


    def get_input_dim(self) -> int:
        return self.input_dim

    def forward(self, embedded_input: torch.Tensor, input_mask: torch.Tensor, fertilities = None) -> torch.Tensor:
        """
        compute log probs for output. Runs an LSTM over the "fertility dimension".
        :param input_mask: shape (batch_size, input_seq_len)
        :param embedded_input: shape (batch_size, input_seq_len, embedding_dim)
        :return: shape (batch_size, input_seq_len, maximum fertility + 1, vocab size)
        """

        batch_size, input_seq_len, embedding_dim = embedded_input.shape

        reshaped_embedded_input = embedded_input.reshape([batch_size * input_seq_len, embedding_dim])

        decoder_hidden = torch.zeros((batch_size*input_seq_len, self.hidden_dim), device=embedded_input.device)
        decoder_context = torch.zeros_like(decoder_hidden)
        last_output = torch.zeros((batch_size * input_seq_len, self.vocab_size), device=embedded_input.device)
        last_output[:, self.oov_id] = 1.0 # use OOV symbol as initial "last output"

        if self.range_tensor is None or self.range_tensor.shape != (batch_size*input_seq_len, self.vocab_size):
            self.range_tensor = torch.arange(self.vocab_size, device=embedded_input.device).repeat(batch_size*input_seq_len).reshape(
                (batch_size*input_seq_len, self.vocab_size))

        dropout_mask = F.dropout(torch.ones_like(decoder_hidden), self.dropout, self.training, inplace=False)

        outputs = []
        for k in range(self.maximum_fertility+1):

            if self.training or self.soft_at_test:
                if self.training and self.use_gumbel_noise:
                    per_sample_weights = F.gumbel_softmax(torch.log(last_output), tau=1.0, dim=-1, hard=self.hard)
                else:
                    per_sample_weights = last_output
                embedded_last_step = F.embedding_bag(self.range_tensor, self.embedding_weight.weight, per_sample_weights=per_sample_weights,
                                                     mode="sum")
            else:
                embedded_last_step = self.embedding_weight(torch.argmax(last_output, dim=-1))

            lstm_input = torch.cat([reshaped_embedded_input, embedded_last_step], dim=1)
            decoder_hidden, decoder_context = self.lstm(
                lstm_input, (decoder_hidden, decoder_context)
            )
            decoder_hidden *= dropout_mask
            # F.gumbel_softmax()
            activations = torch.log_softmax(self.output_layer(decoder_hidden), dim=-1)
            last_output = activations.exp() #shape (batch_size * input_seq_len, vocab)
            outputs.append(activations)

        stacked = torch.stack(outputs, dim=1)

        # Separate batch and input seq len dimension:
        return stacked.reshape([batch_size, input_seq_len, self.maximum_fertility+1, self.vocab_size])
