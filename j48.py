import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier
import weka.plot.graph as graph  # for graph plotting

# Start the JVM (with the default options)
jvm.start()

# Load a dataset
loader = Loader("weka.core.converters.CSVLoader")
data = loader.load_file("your_dataset.csv")
data.class_is_last()  # assuming the class attribute is the last attribute

# Build a J48 classifier
cls = Classifier(classname="weka.classifiers.trees.J48")
cls.build_classifier(data)

# Plot the J48 tree
graph.plot_dot_graph(cls.graph)

# Stop the JVM
jvm.stop()