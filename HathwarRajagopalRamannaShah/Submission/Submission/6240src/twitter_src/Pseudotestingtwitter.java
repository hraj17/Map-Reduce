/**
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package cs6240.project.decisiontree;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.net.URI;
import cs6240.project.decisiontree.Histogram.InitializationPartioner;

public class Pseudotestingtwitter {

    public static class TestingMapper extends
            Mapper<Object, Text, IntWritable, IntWritable> {

        // hashmap to maintain the histograms
        HashMap metadata = new HashMap();
        // misc variables
        float spam_ratio = 0;
        float nspam_ratio = 0;
        Integer Correct_counter = 0;
        Integer Incorrect_counter = 0;
        Integer Bucket_count = 5;

        /*
         * the part files from the training model is aggregated into a single file
         * which will now contain histogram metadata about all the attributes
         * this is on S3 and has to be loaded onto the testing machines using distributed
         * cache. The contents are then read and the hash of histograms indexed 
         * by attribute index is constructed. 
         */
        public void setupMetadata(Context context) {

            int att;
            String[] input = null;
            String sCurrentLine;
            String[] flatHistogram = null;
            BufferedReader br = null;

            try {
                Path[] uris = DistributedCache.getLocalCacheFiles(context
                        .getConfiguration());

                for (int i = 0; i < uris.length; i++) {
                    if (uris[i].toString().contains("metadata")) {

                        String chunk = null;
                        br = new BufferedReader(new FileReader(
                                uris[i].toString()));
                    }
                }

                //populating the histogram
                br = new BufferedReader(new FileReader(uris[0].toString()));
                while ((sCurrentLine = br.readLine()) != null) {
                    int[][] his = new int[4][Bucket_count];
                    for (int[] temp : his)
                        Arrays.fill(temp, 0);
                    input = sCurrentLine.split("\t");
                    att = Integer.parseInt((input[0]));

                    flatHistogram = input[1].split(", ");
                    for (String hi : flatHistogram) {
                        String[] parameter = hi.split(" ");
                        int i = Integer.parseInt(parameter[0]);
                        int j = Integer.parseInt(parameter[1]);
                        his[i][j] = Integer.parseInt(parameter[2]);
                    }
                    metadata.put(att, his);

                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    if (br != null)
                        br.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            }

        }

        /*
         * each dataset has a unique ratio of spam and nonspam users
         * which is required for the naive bayes formula and hence
         * is calculated here
         */
        public void calculateSpamRatio() {
            Integer GrandTotal = 0;
            Integer SpamTotal = 0;
            Integer NspamTotal = 0;
            for (int i = 0; i < 15; i++) {
                int[][] his = (int[][]) (metadata.get(i));
                for (int j = 0; j < Bucket_count; j++) {
                    GrandTotal += his[1][j];
                    SpamTotal += his[2][j];
                    NspamTotal += his[3][j];
                }
            }
            spam_ratio = (float) SpamTotal / (float) GrandTotal;
            nspam_ratio = (float) NspamTotal / (float) GrandTotal;

        }

        protected void setup(Context context) throws IOException,
                InterruptedException {
            setupMetadata(context);
            calculateSpamRatio();
        }

        /*
         * the testing map iterates over each record and fits each attribute value 
         * into a bucket calculating the probability for it based on the label
         * the total product of probabilitiies over all attributes results in 2
         * values, comparing which will help us determine the label for that record
         * This is then compared to the records label and if its a match we increment 
         * a"correct" counter else an "incorrect" counter and send them both to the reducer
         */
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] records = value.toString().split("\n");
            for (String record : records) {
                String[] attributes = record.trim().split(",");

                float SPAM = spam_ratio;
                float NONSPAM = nspam_ratio;
                int prediction = 0;
                Integer label = null;
                Integer container_bucket = null;
                for (int i = 0; i < attributes.length - 1; i++) {
                    Integer attrib = Integer.parseInt(attributes[i]);
                    label = Integer.parseInt(attributes[attributes.length - 1]);
                    SPAM = 1;
                    NONSPAM = 1;

                    int[] buckets = ((int[][]) metadata.get(i))[0];
                    // =======get bucket number that contains attr=============
                    for (int col = 0; col < Bucket_count; col++) {
                        if (attrib <= buckets[col]) {
                            container_bucket = col;

                            Integer TC = (((int[][]) metadata.get(i))[1][container_bucket]);
                            Integer SC = (((int[][]) metadata.get(i))[2][container_bucket]);
                            Integer NSC = (((int[][]) metadata.get(i))[3][container_bucket]);

                            // ===calculating probability
                            SPAM *= (float) SC / (float) TC;
                            NONSPAM *= (float) NSC / (float) TC;

                            break;
                        }
                    }

                }
                if (SPAM > NONSPAM)
                    prediction = 1;
                else
                    prediction = 0;
                if (label == prediction)
                    Correct_counter++;
                else
                    Incorrect_counter++;

            }
        }

        protected void cleanup(Context context) throws IOException,
                InterruptedException {

            super.cleanup(context);
            context.write(new IntWritable(1), new IntWritable(Correct_counter));
            context.write(new IntWritable(0),
                    new IntWritable(Incorrect_counter));
            // DistributedCache.purgeCache(context.getConfiguration());
        }

    }

    /*
     * the partition happen over 2 reduce tasks based on the binary values
     * for correct and incorrect counters
     */
    public static class TestingPartioner extends
            Partitioner<IntWritable, IntWritable> {
        @Override
        public int getPartition(IntWritable key, IntWritable value,
                int numofpartitions) {

            return Math.abs(key.hashCode() % numofpartitions);
        }
    }

    /*
     * The reducer simple iterates over the values of each such counter and 
     * sums it up and outputs it
     */
    public static class TestingReducer extends
            Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {

        Integer sum = 0;
        Integer k = 0;

        public void reduce(IntWritable key, Iterable<IntWritable> values,
                Context context) throws IOException, InterruptedException {

            k = key.get();
            for (IntWritable val : values) {

                sum += val.get();
            }
        }

        protected void cleanup(Context context) throws IOException,
                InterruptedException {
            super.cleanup(context);
            context.write(new IntWritable(k), new IntWritable(sum));
        }

    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args)
                .getRemainingArgs();
        DistributedCache.addCacheFile(new URI(
                "s3://hr6240/histogram/5/metadata5"), conf);
        Job job = new Job(conf, "word count");
        job.setJarByClass(Pseudotestingtwitter.class);
        job.setMapperClass(TestingMapper.class);
        job.setReducerClass(TestingReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setPartitionerClass(TestingPartioner.class);
        job.setNumReduceTasks(2);
        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}