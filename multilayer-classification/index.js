const PREDICTION_ELEMENT = document.getElementById('prediction')
const CANVAS = document.getElementById('canvas')
const CTX = canvas.getContext('2d')

import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js'

// Grab a reference to the MIST input values (pixel data).
const INPUTS = TRAINING_DATA.inputs;
// Grab reference to the MIST output values.
const OUTPUTS = TRAINING_DATA. outputs;

console.log(TRAINING_DATA)

// Shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature Array is 2 dimensional.
const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// Output feature Array is 1 dimensional.
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

const model = tf.sequential()

model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}))

model.add(tf.layers.dense({units: 16, activation: 'relu'}))

model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

model.summary()

train()

async function train() {
  console.log('training...')
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  })
  
  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true,
    validationSplit: 0.2,
    batchSize: 512,
    epochs: 50,
    callbacks: {onEpochEnd: logProgress}
  })
  
  INPUTS_TENSOR.dispose()
  OUTPUTS_TENSOR.dispose()
  
  evaluate()
}

function evaluate() {
  console.log('evaluating...')
  const OFFSET = Math.floor(Math.random() * INPUTS.length)
  
  const answer = tf.tidy(function() {
    const newInput = tf.tensor1d(INPUTS[OFFSET])
    
    const output = model.predict(newInput.expandDims())
    output.print()
    
    return output.squeeze().argMax()
  })
  
  answer.array().then(function(index) {
    PREDICTION_ELEMENT.innerText = index
    PREDICTION_ELEMENT.className = index === OUTPUTS[OFFSET] ? 'correct' : 'wrong'
    
    answer.dispose()
    
    drawImage(INPUTS[OFFSET])
  })
}


function logProgress(epoch, logs) {
  console.log(epoch)
}


function drawImage(digit) {
  const imageData = CTX.getImageData(0,0,28,28)
  
  for (let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255 // R
    imageData.data[i * 4 + 1] = digit[i] * 255 // G
    imageData.data[i * 4 + 2] = digit[i] * 255 // B
    imageData.data[i * 4 + 3] = digit[i] * 255 // ALPHA
  }
  
  CTX.putImageData(imageData, 0, 0)
  
  setTimeout(evaluate, 3000)
}