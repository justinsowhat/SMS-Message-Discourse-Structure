import opennlp.maxent.GIS;
import opennlp.model.Event;
import opennlp.model.EventStream;
import opennlp.model.MaxentModel;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Created by justinso on 6/17/16.
 */
public class MaxEntModel {

    public static void main(String[] args) throws Exception{

        String dataFileName = "MaxEntModel/src/main/resources/data.txt";
        Scanner scanner = new Scanner(new File(dataFileName));
        List<Event> data = new ArrayList<Event>();
        while (scanner.hasNextLine()){
            String line = scanner.nextLine();
            String[] tokens = line.split("\\s+");
            String outcome = tokens[0];
            String[] features = Arrays.copyOfRange(tokens, 1, tokens.length);
            Event event = new Event(outcome, features);
            data.add(event);
        }

        Double test_size = data.size() * 0.2;
        int test_size_int = test_size.intValue();
        int k = 0;
        int nFold = 1;
        float correct = 0;
        float total = 0;
        while (k < nFold) {

            List<Event> train_set = new CopyOnWriteArrayList<Event>();
            List<Event> test_set = new CopyOnWriteArrayList<Event>();

            //if (k < (nFold - 1)) {
                test_set.addAll(data.subList(k * test_size_int, (k + 1) * test_size_int));
                train_set.addAll(data.subList((k + 1) * test_size_int, data.size()));
                train_set.addAll(data.subList(0, k * test_size_int));

//            } else {
//                test_set.addAll(data.subList(k * test_size_int, data.size()));
//                test_set.addAll(data.subList(0, test_size_int));
//                train_set.addAll(data.subList(test_size_int, k * test_size_int));

            //}
            EventStream es = new ListEventStream(train_set);
            MaxentModel trainedModel = GIS.trainModel(es, 150000, 0, false, true);

            for (Event e: test_set){
                String outcome = e.getOutcome();
                if (outcome.equals("outcome=1")) {
                    String[] features = e.getContext();
                    double[] outcomeProbs = trainedModel.eval(features);
                    String modelOutcome = trainedModel.getBestOutcome(outcomeProbs);
                    if (outcome.equals(modelOutcome)) {
                        correct++;
                    }
                    total++;
                }
            }
            k++;
        }
        System.out.println("number of lines: " + data.size());
        System.out.println("The accuracy of the system is " + ((correct/total) * 100) + "%.");
    }


}
