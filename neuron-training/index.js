import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js'

const INPUTS = TRAINING_DATA.inputs

const OUTPUTS = TRAINING_DATA.outputs

console.log(TRAINING_DATA)

tf.util.shuffleCombo(INPUTS, OUTPUTS)

const INPUTS_TENSOR = tf.tensor2d(INPUTS)

const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS)

function normalize(tensor, min, max) {
  const result = tf.tidy(function () {
    const MIN_VALUES = min || tf.min(tensor, 0)
    
    const MAX_VALUES = max || tf.max(tensor, 0)
    
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES)
    
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES)
    
    const NORMALIZED_VALLUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE)
    
    return {NORMALIZED_VALLUES, MIN_VALUES, MAX_VALUES}
  })
  return result
}

const FEATRED_RESULTS = normalize(INPUTS_TENSOR)

INPUTS_TENSOR.dispose()

FEATRED_RESULTS.NORMALIZED_VALLUES.print()

FEATRED_RESULTS.MIN_VALUES.print()

FEATRED_RESULTS.MAX_VALUES.print()

const model = tf.sequential()

model.add(tf.layers.dense({inputShape: [2], units: 1}))

model.summary()

async function train() {
  const LEARNING_RATE = 0.01
  
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError'
  })
  
  let results = await model.fit(FEATRED_RESULTS.NORMALIZED_VALLUES, OUTPUTS_TENSOR, {
    validationSplit: 0.15,
    shuffle: true,
    batchSize: 64,
    epochs: 10,
  })
  
  OUTPUTS_TENSOR.dispose()
  FEATRED_RESULTS.NORMALIZED_VALLUES.dispose()
  
  console.log('Average error loss : ' + Math.sqrt(results.history.loss[results.history.loss.length - 1]))
  console.log('Average validation error loss : ' + Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]))
  
  evaluate()
}

function evaluate() {
  tf.tidy(function () {
    const newInput = normalize(tf.tensor2d([[750, 1]]), FEATRED_RESULTS.MIN_VALUES, FEATRED_RESULTS.MAX_VALUES)
    
    const output = model.predict(newInput.NORMALIZED_VALLUES)
    output.print()
  })
  
  FEATRED_RESULTS.MIN_VALUES.dispose()
  FEATRED_RESULTS.MAX_VALUES.dispose()
  model.dispose()
  
  console.log(tf.memory().numTensors)
}

train()
