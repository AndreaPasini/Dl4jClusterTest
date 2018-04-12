package datasets;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.IOException;

/**
 * Iterates between the files of a directory.
 */
public class DirectoryIterator {
    private volatile RemoteIterator<LocatedFileStatus> fileIterator;

    /**
     * Constructor.
     * @param sc Spark context
     * @param path directory (e.g. "./data/dir1" for local directory, "hdfs://BigDataHA/user/user1/data/" for hdfs)
     */
    public DirectoryIterator(JavaSparkContext sc, String path) throws IOException {
        Configuration conf = sc.hadoopConfiguration();
        FileSystem fs = org.apache.hadoop.fs.FileSystem.get(conf);
        fileIterator = fs.listFiles(new Path(path), true);
    }

    /**
     * Get next file path.
     */
    public Path nextFile() throws IOException {
        return fileIterator.next().getPath();
    }

    /**
     * @return true if more files are available.
     */
    public boolean hasNext() throws IOException {
        return fileIterator.hasNext();
    }
}
