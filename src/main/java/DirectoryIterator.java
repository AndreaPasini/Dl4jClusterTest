import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.FileNotFoundException;
import java.io.IOException;

public class DirectoryIterator {
    public volatile RemoteIterator<LocatedFileStatus> hdfsIterator;

    protected void initIterator(String hdfsUrl, JavaSparkContext sc) throws IOException {
        Configuration conf = sc.hadoopConfiguration();
        FileSystem fs = org.apache.hadoop.fs.FileSystem.get(conf);
        hdfsIterator = fs.listFiles(new Path(hdfsUrl), true);
    }

}
