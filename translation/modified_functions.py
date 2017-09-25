from tensorflow.python.ops import variable_scope
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import embedding_attention_decoder

from tensorflow.contrib import legacy_seq2seq


def embedding_attention_seq2seq(
		encoder_inputs,
		decoder_inputs,
		cell,
		num_encoder_symbols,
		num_decoder_symbols,
		embedding_size,
		num_heads=1,
		output_projection=None,
		feed_previous=False,
		dtype=None,
		scope=None,
		initial_state_attention=False):

	with variable_scope.variable_scope(scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
		dtype = scope.dtype
		# Encoder.
		#####################
		# modified deepcopy #
		#####################
		# encoder_cell = copy.deepcopy(cell)
		encoder_cell = core_rnn_cell.EmbeddingWrapper(
			# encoder_cell,
			cell,
			embedding_classes=num_encoder_symbols,
			embedding_size=embedding_size
		)
		encoder_outputs, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)

		# First calculate a concatenation of encoder outputs to put attention on.
		top_states = [
			array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
		]
		attention_states = array_ops.concat(top_states, 1)

		# Decoder.
		output_size = None
		if output_projection is None:
			cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
			output_size = num_decoder_symbols

		if isinstance(feed_previous, bool):
			return embedding_attention_decoder(
				decoder_inputs,
				encoder_state,
				attention_states,
				cell,
				num_decoder_symbols,
				embedding_size,
				num_heads=num_heads,
				output_size=output_size,
				output_projection=output_projection,
				feed_previous=feed_previous,
				initial_state_attention=initial_state_attention)

		# If feed_previous is a Tensor, we construct 2 graphs and use cond.
		def decoder(feed_previous_bool):
			reuse = None if feed_previous_bool else True
			with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=reuse):
				print variable_scope.get_variable_scope()._name, variable_scope.get_variable_scope()._name_scope
				outputs, state = embedding_attention_decoder(
					decoder_inputs,
					encoder_state,
					attention_states,
					cell,
					num_decoder_symbols,
					embedding_size,
					num_heads=num_heads,
					output_size=output_size,
					output_projection=output_projection,
					feed_previous=feed_previous_bool,
					update_embedding_for_previous=False,
					initial_state_attention=initial_state_attention)
				state_list = [state]
				if nest.is_sequence(state):
					state_list = nest.flatten(state)
				return outputs + state_list

		outputs_and_state = control_flow_ops.cond(
			feed_previous, lambda: decoder(True), lambda: decoder(False)
		)
		outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
		state_list = outputs_and_state[outputs_len:]
		state = state_list[0]
		if nest.is_sequence(encoder_state):
			state = nest.pack_sequence_as(structure=encoder_state, flat_sequence=state_list)
		return outputs_and_state[:outputs_len], state


legacy_seq2seq.embedding_attention_seq2seq = embedding_attention_seq2seq

