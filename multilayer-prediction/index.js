const INPUTS = []
for (let n = 1; n <= 20; n++) INPUTS.push(n)

const OUTPUTS = []
for (let n = 0; n < INPUTS.length; n++) OUTPUTS.push(INPUTS[n] * INPUTS[n])

console.log(OUTPUTS)

const INPUTS_TENSOR = tf.tensor1d(INPUTS)

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

model.add(tf.layers.dense({inputShape: [1], units: 25, activation: 'relu'}))

model.add(tf.layers.dense({units: 5, activation: 'relu'}))


model.add(tf.layers.dense({units: 1}))

model.summary()

const LEARNING_RATE = 0.0001
const OPTIMIZER = tf.train.sgd(LEARNING_RATE)

async function train() {  
  model.compile({
    optimizer: OPTIMIZER,
    loss: 'meanSquaredError'
  })
  
  let results = await model.fit(FEATRED_RESULTS.NORMALIZED_VALLUES, OUTPUTS_TENSOR, {
    callbacks: {onEpochEnd: logProgress},
    shuffle: true,
    batchSize: 2,
    epochs: 200,
  })
  
  OUTPUTS_TENSOR.dispose()
  FEATRED_RESULTS.NORMALIZED_VALLUES.dispose()
  
  console.log('Average error loss : ' + Math.sqrt(results.history.loss[results.history.loss.length - 1]))
  
  evaluate()
}

function evaluate() {
  tf.tidy(function () {
    const newInput = normalize(tf.tensor1d([7]), FEATRED_RESULTS.MIN_VALUES, FEATRED_RESULTS.MAX_VALUES)
    
    const output = model.predict(newInput.NORMALIZED_VALLUES)
    output.print()
  })
  
  FEATRED_RESULTS.MIN_VALUES.dispose()
  FEATRED_RESULTS.MAX_VALUES.dispose()
  model.dispose()
  
  console.log(tf.memory().numTensors)
}

train()

function logProgress(epoch, logs) {
  console.log(epoch, Math.sqrt(logs.loss))
  if (epoch == 70) OPTIMIZER.setLearningRate(LEARNING_RATE/2)
}
