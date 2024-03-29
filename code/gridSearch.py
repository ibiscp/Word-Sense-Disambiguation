from sklearn.model_selection import ParameterGrid
import tensorflow.keras as K
import random
import numpy as np
import math
from random import shuffle

class gridSearch:

    def __init__(self, build_fn, param_grid, vocab_size, sentence_size, output_size):
        self.build_fn = build_fn
        self.param_grid = param_grid
        self.best_score = 0
        self.best_params = None
        self.results = []
        self.vocab_size = vocab_size
        self.sentence_size = sentence_size
        self.output_size = output_size
        self.iter = 0

    def fit(self, X, y, X_test, y_test):

        for g in ParameterGrid(self.param_grid):

            self.iter += 1
            print('\nTraining:', str(self.iter) + '/' + str(len(ParameterGrid(self.param_grid))), '- Parameters:', g)

            # Model
            model = self.build_fn(vocab_size=self.vocab_size, sentence_size=self.sentence_size, output_size=self.output_size, mergeMode=g['mergeMode'], lstmLayers=g['lstmLayers'], embedding_size=g['embedding_size'])

            # model.summary()

            # Callback
            callback_str = '_'.join(['%s-%s' % (key, str(value)) for (key, value) in g.items()])
            cbk = K.callbacks.TensorBoard("../resources/logging/" + callback_str)

            # Fit generator
            model.fit_generator(self.generator(X, y, batch_size=g['batchSize'], output_size=self.output_size), steps_per_epoch=math.ceil(len(X) / g['batchSize']), validation_data=(X_test, y_test), epochs=g['epochs'], callbacks=[cbk])

            print('\tEvaluating')
            loss, acc = model.evaluate(X_test, y_test, verbose=1)
            print('\tLoss: %f - Accuracy: %f' % (loss, acc))

            self.results.append({'loss':loss, 'acc':acc, 'params':g})

            # Save model
            print("\tSaving model")
            model.save("../resources/model.h5")

            # Write to results
            with open('../resources/results.txt', "a+") as f:
                f.write("Loss: %f - Accuracy: %f - Parameters: %r\n" % (loss, acc, g))

            if acc > self.best_score:
                self.best_score = acc
                self.best_params = g

                # Save model
                print("\tSaving model")
                model.save("../resources/model.h5")

                # Substitute
                with open("../resources/results.txt") as f:
                    lines = f.readlines()

                lines[0] = "Best: " + self.best_score + " using " + self.best_params + "\n"

                with open("../resources/results.txt", "w+") as f:
                    f.writelines(lines)

    def generator(self, features, labels, batch_size, output_size):
        # batch_features = np.zeros((batch_size, self.sentence_size))
        # batch_labels = np.zeros((batch_size, self.sentence_size, output_size))
        #
        # indexes = shuffle(range(len(features)))

        ids = np.arange(features.shape[0])
        np.random.shuffle(ids)
        l = len(ids)

        for batch in range(0, l, batch_size):
            batch_ids = ids[batch:min(batch + batch_size, l)]
            yield features[batch_ids], labels[batch_ids]


        # for i in range(batch_size):
        #     # choose random index in features
        #     index = random.randint(0, len(features)-1)
        #     batch_features[i] = features[index]
        #     batch_labels[i] = labels[index]
        # yield batch_features, batch_labels


    def summary(self):
        # Summarize results
        print('\nSummary')
        print("Best: %f using %s" % (self.best_score, self.best_params))
        for res in self.results:
            print("Loss: %f - Accuracy: %f - Parameters: %r" % (res['loss'], res['acc'], res['params']))