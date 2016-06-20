import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.Random;


public class NeuralNetworkModel {

    private static Logger log = LoggerFactory.getLogger(NeuralNetworkModel.class);
    // RNN dimensions
//    public static final int HIDDEN_LAYER_WIDTH = 50;
//    public static final int HIDDEN_LAYER_CONT = 2;
//    public static final Random r = new Random(7894);
    private static final String FILENAME = "data.txt";

    public static void main(String[] args) throws  Exception {

        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource(FILENAME).getFile()));

        int labelIndex = 0;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 2;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 15192;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);

        DataSet next = iterator.next();

        final int numInputs = 8;
        int outputNum = 2;
        int iterations = 100000;
        long seed = 6;


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .activation("tanh")
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.1)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(5)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(5).nOut(5)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation("softmax")
                .nIn(5).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1000));

        //Normalize the full data set. Our DataSet 'next' contains the full 150 examples
        next.normalizeZeroMeanZeroUnitVariance();
        //next.shuffle();
        //split test and train
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(.80);  //Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        model.fit(trainingData);

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(2);
        DataSet test = testAndTrain.getTest();
        INDArray output = model.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        System.out.println(eval.stats());
        System.out.println("F1 on label 1: " + eval.f1(1));
        System.out.println("Precision on label 1: " + eval.precision(1));
        System.out.println("Recall on label 1: " + eval.recall(1));
    }

}

