import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaSparkContext;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

public class DatasetBuilder {

    public static HashMap<String, INDArray> readClassLabels(JavaSparkContext sc, String path) throws IOException {
        HashMap<String, INDArray> classLabels=new HashMap<>();
        FileInputStream fi;
        DataInputStream dis = getFileStream(sc, path);
        BufferedReader br = new BufferedReader(new InputStreamReader(dis));

        String line;
        List<String> labelList = new LinkedList<>();
        while ((line=br.readLine())!=null)
            labelList.add(line);
        int i=0;
        //Generate 1-hot vectors
        for (String label : labelList) {
            float[] label1Hot = new float[labelList.size()];
            label1Hot[i] = 1;
            classLabels.put(label, Nd4j.create(label1Hot));
            i++;
        }
        return classLabels;
    }

    public static DataInputStream getFileStream(JavaSparkContext sc, String path) throws IOException {
        //Get filesystem object
        Configuration conf = sc.hadoopConfiguration();
        FileSystem fs = org.apache.hadoop.fs.FileSystem.get(conf);

        //Return DataInputStream
        return fs.open(new Path(path));
    }

}
