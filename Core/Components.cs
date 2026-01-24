using System;
using System.Reflection.Emit;

namespace NEAT_GNN.Core
{
    public enum NodeType
    {
        Input,
        Hidden,
        Output,
        Bias
    }

    public class Node
    {
        public int Id;
        public NodeType Type;
        public int Layer;
        public Func<double, double> Activation;

        public double Input;
        public double Output;

        public Node(int id, NodeType type, int layer, Func<double, double> activation)
        {
            Id = id;
            Type = type;
            Layer = layer;
            Activation = activation;
            Input = 0.0;
            Output = 0.0;
        }

        public void Activate()
        {
            if (Activation != null)
            {
                Output = Activation(Input);
            }
            else
            {
                Output = 0.0;
            }
            Input = 0.0; // Reset
        }
    }

    public class Connection
    {
        public Node InNode;
        public Node OutNode;

        public int InnovationNumber;
        public double Weight;
        public bool Enabled;

        public Connection(Node inNode, Node outNode, int innvNumber, double weight, bool enabled)
        {
            InNode = inNode;
            OutNode = outNode;
            InnovationNumber = innvNumber;
            Weight = weight;
            Enabled = enabled;
        }
    }
}
