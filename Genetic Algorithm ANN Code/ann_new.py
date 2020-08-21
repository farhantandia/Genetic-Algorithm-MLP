import numpy


class ANN:

    @classmethod
    def sigmoid(self, inpt):
        return 1.0/(1.0+numpy.exp(-1*inpt))
    @classmethod
    def relu(self, inpt):
        result = inpt
        result[inpt<0] = 0
        return result

    @classmethod
    def predict_outputs(self,weights_mat, data_inputs, data_outputs, activation="relu"):
        predictions = numpy.zeros(shape=(data_inputs.shape[0]))
        for sample_idx in range(data_inputs.shape[0]):
            r1 = data_inputs[sample_idx, :]
            for curr_weights in weights_mat:
                r1 = numpy.matmul(r1, curr_weights)
                if activation == "relu":
                    r1 = self.relu(r1)
                elif activation == "sigmoid":
                    r1 = self.sigmoid(r1)
            predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
            predictions[sample_idx] = predicted_label
        correct_predictions = numpy.where(predictions == data_outputs)[0].size
        accuracy = (correct_predictions/data_outputs.size)*100
        return accuracy, predictions
    @classmethod
    def fitness(self,weights_mat, data_inputs, data_outputs, activation="relu"):
        accuracy = numpy.empty(shape=(weights_mat.shape[0]))
        for sol_idx in range(weights_mat.shape[0]):
            curr_sol_mat = weights_mat[sol_idx, :]
            accuracy[sol_idx], _ = self.predict_outputs(curr_sol_mat, data_inputs, data_outputs, activation=activation)
        return accuracy

