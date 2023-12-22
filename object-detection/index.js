const MODEL_PATH = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json'
const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');


// Store the resulting model in the global scope of our app.
let model = undefined;

(async () => {
  const model = await tf.loadLayersModel(MODEL_PATH);
  model.summary();
  
  // create a batch of 1
  const input = tf.tensor2d([[870]]);

  // create a batch of 3
  const inputBatch = tf.tensor2d([[500],[1100],[970]]);

  // Actually make the predictions for each batch
  const result = model.predict(input);
  const resultBatch = model.predict(inputBatch);

  // print results to the ocnsole
  result.print(); // or use .array() to get results back as array
  resultBatch.print(); // or use .array() to get results back as array

  input.dispose();
  inputBatch.dispose();
  result.dispose();
  resultBatch.dispose();
  model.dispose();
})();