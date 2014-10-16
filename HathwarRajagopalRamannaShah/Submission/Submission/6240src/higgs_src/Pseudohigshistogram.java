package cs6240.project.decisiontree;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class Pseudohigshistogram {
    public static int attr_count = 28;
    //Arrays to keep track of the minimum and maximum value encountered.
    public static ArrayList<Double> Attr_Min = new ArrayList<Double>(attr_count);
    public static ArrayList<Double> Attr_Max = new ArrayList<Double>(attr_count);

    public static class InitilizationMapper extends
            Mapper<Object, Text, Text, Text> {
        /*
         * A hashmap to have the attribute index as key and another hashmap 
         * as value the secondary hash's key is a combination of label 
         * and attribute value with the frequency of that combination as value
         */
        private static HashMap<Integer, HashMap<String, Integer>> attributeHash;

        @Override
        protected void setup(Context context) throws IOException,
                InterruptedException {
            super.setup(context);
            attributeHash = new HashMap<Integer, HashMap<String, Integer>>();

            for (int i = 0; i < attr_count; i++) {
                Pseudohigshistogram.Attr_Min.add(i, Double.MAX_VALUE);
            }
            for (int i = 0; i < attr_count; i++) {
                Pseudohigshistogram.Attr_Max.add(i, Double.MIN_VALUE);
            }

        }


        @Override
        protected void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] records = value.toString().split("\n");
            for (String record : records) {
                String[] attributes = record.trim().split(",");
                for (int i = 1; i < attributes.length; i++) {

                    Double attrib = Double.parseDouble(attributes[i]);
                    String label = attributes[0];
                    String attrib_value = attrib + "," + label;

                    /*
                     * updates the min and max arrays for the attributes
                     */
                    if (Pseudohigshistogram.Attr_Min.get(i - 1) > attrib)
                        Pseudohigshistogram.Attr_Min
                                .set(i - 1, (Double) attrib);
                    if (Pseudohigshistogram.Attr_Max.get(i - 1) < attrib)
                        Pseudohigshistogram.Attr_Max
                                .set(i - 1, (Double) attrib);

                    /*
                     * based on the attribute index and the label and value 
                     * combination string resective counters are updated 
                     * to keep track of frequency.
                     */
                    if (attributeHash.containsKey(i - 1)) {
                        HashMap<String, Integer> attribValueMap = attributeHash
                                .get(i - 1);
                        // Searching for column number
                        if (attribValueMap.containsKey(attrib_value)) {
                            int count = attribValueMap.get(attrib_value);
                            attribValueMap.put(attrib_value, ++count);
                        } else
                            attribValueMap.put(attrib_value, 1);
                        attributeHash.put(i - 1, attribValueMap);
                    } else {
                        HashMap<String, Integer> newAttribValueMap = new HashMap<String, Integer>();
                        newAttribValueMap.put(attrib_value, 1);
                        attributeHash.put(i - 1, newAttribValueMap);

                    }
                }
            }

        }

        @Override
        protected void cleanup(Context context) throws IOException,
                InterruptedException {
            Text attributeDetails = new Text();
            Text attrValueCount = new Text();
            super.cleanup(context);
            //each hashmap entry is unique and has to be sent out with the respective
            //frequency to the reducer
            for (Entry<Integer, HashMap<String, Integer>> entry : attributeHash
                    .entrySet()) {
                Integer attrIndex = entry.getKey();
                HashMap<String, Integer> attValueMap = entry.getValue();
                //the index+value+label is the key and the value+frequqency the value

                for (Entry<String, Integer> innerEntry : attValueMap.entrySet()) {
                    attributeDetails.set(attrIndex.toString() + ","
                            + innerEntry.getKey());
                    attrValueCount.set(innerEntry.getKey().toString()
                            .split(",")[0]
                            + "," + innerEntry.getValue());
                    context.write(attributeDetails, attrValueCount);
                }
                //also sending out the min and max for each attribute by having
                // a unique key such that it bubbles to the top on the 
                //intermediate key list
                attributeDetails.set(attrIndex.toString() + ","
                        + (-Double.MIN_VALUE) + ",-1");
                attrValueCount.set(Attr_Min.get(attrIndex).toString() + ","
                        + Attr_Max.get(attrIndex).toString());
                context.write(attributeDetails, attrValueCount);

            }
        }
    }

    /*
     * partition based on the index alone, hence making each reduce task
     * responsible to process just one attribute for all the records.
    */
    public static class InitializationPartioner extends Partitioner<Text, Text> {
        @Override
        public int getPartition(Text key, Text value, int numofpartitions) {
            String[] itr = key.toString().split(",");
            String attributeIndex = itr[0];
            return Math.abs((Integer.parseInt(attributeIndex))
                    % numofpartitions);

        }
    }

    public static class InitilizationReducer extends
            Reducer<Text, Text, Text, Text> {
        //dimensions of the histogram

        public static int attr_count = 28;
        Integer Bucket_count = 5;
        Integer Constant_count = 4;
        int total_frequency;
        //reconstruct the min and max for the attribute index.

        Double min;
        Double max;

        //construct a histogram which has the thresholds in the first row
        //total count of records in each of the respective buckets
        //count of spam records in each of the respective buckets
        //count of non spam records in each of the respective buckets
        Double[][] histogram_metadata = new Double[Constant_count][Bucket_count];
        Integer Attribute_number = 0;

   

        @Override
        protected void setup(Context context) throws IOException,
                InterruptedException {

            super.setup(context);
            //initialization

            total_frequency = 0;
            min = 0.0;
            max = 0.0;
            for (Double[] temp : histogram_metadata)
                Arrays.fill(temp, 0.0);
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            //the composite key is plit to give usefull fields to work on

            String[] reduce_key = key.toString().split(",");
            Attribute_number = Integer.parseInt(reduce_key[0]);
            Double Attribute_value = Double.parseDouble(reduce_key[1]);
            int Label = (int) Double.parseDouble(reduce_key[2]);

            for (Text val : values) {
                //the first key value pair for any reduce task will satisfy this condition
                //and the min and max for this particular attribute index is  setup
                if (Attribute_value == (-Double.MIN_VALUE) && Label == -1) {
                    min = Double.parseDouble(val.toString().split(",")[0]);
                    max = Double.parseDouble(val.toString().split(",")[1]);
                    //the thresholds for our buckets is also calculated

                    Double interval = ((Double) (max - min) / Bucket_count);

                    histogram_metadata[0][0] = interval;// (Integer)Attr_Min.get(Attribute_number);
                    // histogram is setup with the thresholds as calculated above

                    for (int i = 1; i < Bucket_count - 1; i++) {
                        histogram_metadata[0][i] = histogram_metadata[0][i - 1]
                                + Math.round(interval);
                    }
                    histogram_metadata[0][Bucket_count - 1] = max;

                } else {
                    //value is broken down to get information as well

                    Double attr_value = Double.parseDouble(val.toString()
                            .split(",")[0]);
                    int attr_freq = Integer
                            .parseInt(val.toString().split(",")[1]);
                    //the attribute value is then fit into one of the buckets and 
                    // then based on the label the respective frequency is added onto it
                    for (int i = 0; i < Bucket_count; i++) {
                        if ((attr_value <= histogram_metadata[0][i])
                                && (attr_value != -1)) {

                            histogram_metadata[1][i] += attr_freq;
                            if (Label == 1) {
                                histogram_metadata[2][i] += attr_freq;
                            } else {
                                histogram_metadata[3][i] += attr_freq;
                            }

                            break;
                        }
                    }

                }
            }
        }

        @Override
        /*
         * FInally after every reduce task the histogram is 
         * flattened out to a string and sent out 
         * with the attribute index as key
        */
        protected void cleanup(Context context) throws IOException,
                InterruptedException {
            super.cleanup(context);
            StringBuffer flat_histogram = new StringBuffer("");

            for (int i = 0; i < Constant_count; i++) {
                for (int j = 0; j < Bucket_count; j++) {
                    flat_histogram.append(i + " " + j + " "
                            + histogram_metadata[i][j] + ", ");
                }
            }

            context.write(
                    new Text("" + Attribute_number),
                    new Text(flat_histogram.toString().substring(0,
                            flat_histogram.length() - 2)));

        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args)
                .getRemainingArgs();

        Job job = new Job(conf);
        job.setJarByClass(Pseudohigshistogram.class);
        job.setMapperClass(InitilizationMapper.class);
        /* Setting partitioner class */
        job.setPartitionerClass(InitializationPartioner.class);
        job.setNumReduceTasks(attr_count);
        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.setReducerClass(InitilizationReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

}
