// Project UID db1f506d06d84ab787baf250c265e24e

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <set>
#include <cmath>
#include <sstream>
#include <iomanip>
#include "csvstream.h"

using namespace std;

// Classifier class that learns from training data
// and makes predictions
class Classifier {
    private:
        // whether or not to display debug output
        bool debug;
        // total number of posts
        double total_posts; 
        // number of unique words
        int vocab_size; 
        // column that contains labels
        double label_col; 
        // column that contains content
        double content_col; 
        // pair that stores labels and words
        using Words_labels = pair<string, string>; 
        // map storing number of posts that contain a word
        map<string, double> words_posts; 
        // map storing number of posts that contain a label
        map<string, double> labels_posts; 
        // map storing number of posts that contain a label with a certain word
        map<Words_labels, double> words_labels_posts;

        // stores the correct label for test content
        string c_label;
        // stores the predicted label for test content
        string prediction;

    public:
        Classifier() : debug(false), total_posts(0), vocab_size(0),
                        label_col(0), content_col(-1), c_label(""),
                        prediction("") {}

        // EFFECTS: Returns a set containing the unique "words" in the original
        //          string, delimited by whitespace.
        set<string> unique_words(const string &str) {
        // Fancy modern C++ and STL way to do it
        istringstream source{str};
        return {istream_iterator<string>{source},
                istream_iterator<string>{}};
        }

        // MODIFIES: The debug variable.
        // EFFECTS:  Sets the debug variable to true.
        void debug_true() {
            debug = true;
        }

        // REQUIRES: firstline is a valid stringstream that contains
        //           the keywords "tag" and "content", and all words
        //           are separated by commas.
        // MODIFIES: The label_col and content_col variables.
        // EFFECTS:  Reads through firstline and stores which column
        //           contains the tags and which contains the content.
        void get_cols(stringstream &firstline) {
            double count = 0;
            string line;
            while (getline(firstline, line, ',')) {
                if (line == "tag")
                    label_col = count;
                if (line == "content")
                    content_col = count;
                ++count;
            }
        }

        // MODIFIES: vocab_size
        // EFFECTS:  If the passed in parameter (the amount of tiems a word has been
        //           saved) is 1, increment vocab_size
        void increment_vocab(int size) {
            if (size == 1)
                ++vocab_size;
        }

        // REQUIRES: is is a valid ifstream that is in the csv format
        // MODIFIES: All field variables
        // EFFECTS:  Trains the machine by reading saving frequency of labels/content.
        //           First calls get_cols to store the relevant columns.
        //           Then scans through is line by line, adding labels, words,
        //           and combinations to the relevant maps. The function 
        //           unique_words is used to find all the unique words in a line.
        //           Debug outputs are printed out if the debug variable is true.
        void train(ifstream &is) {
            string line;
            getline(is, line);
            stringstream firstline(line);

            if (debug)
                cout << "training data:";
            
            get_cols(firstline);
            double count = 0;

            while (getline(is, line)) {
                count = 0;
                stringstream current(line);
                string label;

                // Go through every line, saving content separated by commas
                while (getline(current, line, ',')) {
                    // If in a label column, add the label to label map
                    if (count == label_col) {
                        label = line;
                        ++labels_posts[label];

                        if (debug)
                            cout << "\n  label = " << label << ", content = ";
                    }

                    if (count == content_col) {
                        if (debug)
                            cout << line;
                    }

                    // If in a content column, turn content into a set of unique
                    // words and add them to the word map. Every word is also 
                    // combined with the label of that line and added to the 
                    // combination map
                    if (count == content_col) {
                        set<string> words = unique_words(line);

                        for (auto it : words) {
                            ++words_posts[it];
                            increment_vocab(words_posts[it]);

                            Words_labels wl(label, it);
                            ++words_labels_posts[wl];
                        }
                    }
                    ++count;
                }
                ++total_posts;
            }
            
            if (debug)
                cout << "\n";
            cout << "trained on " << total_posts << " examples\n";
            if (debug)
                cout << "vocabulary size = " << vocab_size << "\n\n";
        }

        // EFFECTS: Returns the log-prior probability of label.
        double calc_log_prior(string label) {
            return log(labels_posts[label]/total_posts);
        }

        // EFFECTS: Returns the log-likelihood of a word occuring
        //          given the label. Uses different formulas depending
        //          on if the word ever appears with the label, or if
        //          it ever appeared at all in the training file.
        double calc_log_likelihood(string label, string word) {
            Words_labels wl(label, word);

            if (words_posts[word] == 0)
                return log(1/total_posts);
            else if (words_labels_posts[wl] == 0)
                return log(words_posts[word]/total_posts);
            return log(words_labels_posts[wl]/labels_posts[label]);
        }

        // EFFECTS: If debug is true, prints out the debug statements that
        //          list the classes in the training data, the number of examples 
        //          for each, and their log-prior probabilities.
        void map_priors() {
            if (debug) {
                cout << "classes:";

                for (auto it : labels_posts) {
                        cout << "\n  " << it.first << ", ";
                        cout.precision(10);
                        cout << trunc(labels_posts[it.first]) << " examples";
                        cout << ", log-prior = ";
                        cout.precision(3);
                        cout << calc_log_prior(it.first);
                }
            }
        }

        // EFFECTS: If debug is true, prints out the debug statements that
        //          list for each label and each word that occurs for that label,
        //          the number of posts with that label that contained the word, 
        //          and the log-likelihood of the word given the label.
        void map_log_likelihood() {
            if (debug) {
                cout << "\nclassifier parameters:";

                for (auto it : words_labels_posts) {
                        cout << "\n  " << it.first.first << ":" << it.first.second;
                        cout.precision(10);
                        cout << ", count = " << it.second << ", log-likelihood = ";
                        cout.precision(3);
                        cout << calc_log_likelihood(it.first.first, it.first.second);
                }
                
                cout << "\n";
            }
        }

        // MODIFIES: Parameters count, temp, probability, max, prediction, and label.
        // EFFECTS:  First stores the actual label for the corresponding content,
        //           and prints that out. Then stores the content as a set of unique
        //           words and calculates the probability of that content belonging
        //           to every label we have gathered data on. Saves the label that has
        //           the highest probability.
        void find_probability (int &count, string line, string &temp, double &max) {
            double probability = 0;
                                        
            if (count == label_col) {
                c_label = line;
                cout << "\n  correct = " << c_label << ", ";
            }

            if (count == content_col) {
                set<string> words = unique_words(line);
                temp = line;

                // Iterate through all labels we have data on
                for (auto it1 : labels_posts) {
                    probability = 0;

                    // First add the log-prior probability of the label
                    probability += calc_log_prior(it1.first);

                    // Iterate through all the words in the content and add
                    // the log-likelihood probability of each word-label pair
                    for (auto it2 : words) {
                        probability += calc_log_likelihood(it1.first, it2);
                    }

                    // If the probability is the highest so far, saves that
                    // probability and the corresponding label
                    if (probability > max) {
                        max = probability;
                        prediction = it1.first;
                    }
                }
            }
        }

        // REQUIRES: is is a valid ifstream that is in the csv format.
        // EFFECTS: In similar fashion to the training function, reads through
        //          the testing file line by line and finds the most likely
        //          label based on the content using the find_probability
        //          method. Prints out the prediction, probability, and content
        //          that led to the prediction, saving how many the machine gets
        //          right. At the end the performance is printed.
        void predict(ifstream &is) {
            cout << "\ntest data:";

            string line;
            getline(is, line);
            stringstream firstline(line);

            int count = 0;
            while (getline(firstline, line, ',')) {
                if (line == "tag")
                    label_col = count;
                if (line == "content") {
                    content_col = count;
                }
                ++count;
            }

            int correct = 0;
            int total = 0;
            while (getline(is, line)) {
                count = 0;
                string temp;
                stringstream current(line);
                double max = numeric_limits<double>::lowest();

                while (getline(current, line, ',')) {
                    find_probability(count, line, temp, max);
                    ++count;
                }
                
                // Special case, where if there is no content the machine makes
                // a decision based on the most likely log-prior probability
                if (count == 1) {
                    for (auto it : labels_posts) {
                        if (calc_log_prior(it.first) > max) {
                            max = calc_log_prior(it.first);
                            prediction = it.first;
                        }
                    }
                }

                cout << "predicted = " << prediction << ", ";
                cout << "log-probability score = " << max;
                cout << "\n  content = " << temp << "\n";

                if (prediction == c_label)
                    ++correct;

                ++total;
            }

            cout << "\nperformance: " << correct << " / ";
            cout << total << " posts predicted correctly\n";
        }
};

// EFFECT: Checks that the user inputs appropriate arguments.
bool check_args(int argc, char * argv[]) {
    if ((argc != 3 && argc != 4)) {
        cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
        return true;
    }

    if (argc == 4) {
        string debug = argv[3];
        if (debug != "--debug") {
            cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
            return true;
        }
    }

    return false;
}

// EFFECT: Checks that the user inputted files that can be opened.
bool check_files(char * argv[]) {
    string train = argv[1];
    ifstream fin1(train);
    if (!fin1.is_open()) {
        cout << "Error opening file: " << train << endl;
        return true;
    }
    fin1.close();

    string test = argv[2];
    ifstream fin2(test);
    if (!fin2.is_open()) {
        cout << "Error opening file: " << test << endl;
        return true;
    }
    fin2.close();

    return false;
}

// Driver
int main(int argc, char * argv[]) {
    if (check_args(argc, argv))
        return 1;
    if (check_files(argv))
        return 2;

    Classifier *c = new Classifier;

    if (argc == 4)
        c->debug_true();

    ifstream fin1(argv[1]);
    c->train(fin1);
    fin1.close();

    cout.precision(3);

    c->map_priors();
    c->map_log_likelihood();

    ifstream fin2(argv[2]);
    c->predict(fin2);
    fin1.close();

    delete c;
}