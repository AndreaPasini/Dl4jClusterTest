package utils;

import org.apache.commons.compress.compressors.FileNameUtil;
import org.joda.time.DateTime;
import org.joda.time.format.DateTimeFormat;

import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.commons.io.FilenameUtils;

/**
 * Created by francescoventura on 12/04/18.
 */
public class Utils {

    public static String getOutFolderName(String prefix){
        return String.format("%s_out_%s",
                prefix,
                DateTime.now().toString(DateTimeFormat.forPattern("yyyyMMddHHmmss"))
        );
    }

    public static String getFileNameFromURI(String path){
        Path p = Paths.get(path);
        return FilenameUtils.removeExtension(p.getFileName().toString());
    }
}
