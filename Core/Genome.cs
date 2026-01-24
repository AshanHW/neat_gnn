using NEAT_GNN.Utilities;
using System;
using System.Diagnostics;


namespace NEAT_GNN.Core
{
    public enum WeightInitMode
    {
        RandomUniform,
        RandomNormal,
        Zero,
        Xavier
    }
    public class Genome
    {
        public int ID;
        public double RawFitness;
        public double AdjustedFitness;
        public Dictionary<int, Node> Nodes;
        public List<Connection> Connections;
        public List<int> ExecutionOrder;
        public Dictionary<int, List<Connection>> outgoing;

        public Genome(int id, Dictionary<int, Node> nodes, List<Connection> connections)
        {
            ID = id;
            Nodes = nodes;
            Connections = connections;
            ExecutionOrder = new List<int>();
            outgoing = new();
            AdjustedFitness = 0;
        }

        public void InitialiseWeights(Random rng, WeightInitMode mode)
        {
            foreach (Connection conn in Connections)
            {
                if (!conn.Enabled) continue;

                conn.Weight = mode switch
                {
                    WeightInitMode.RandomUniform => rng.NextDouble() * 2 - 1,
                    WeightInitMode.RandomNormal => Utils.NextGaussian(rng, 0, 1),
                    WeightInitMode.Zero => 0.0,
                    WeightInitMode.Xavier => Utils.NextGaussian(rng, 0, Math.Sqrt(2.0 / Nodes.Count)),
                    _ => conn.Weight
                };
            }
        }

        public void AddNode(Node node)
        {
            Nodes.Add(node.Id, node);
        }

        public void AddConnection(Connection conn)
        {
            Connections.Add(conn);
        }

        public void TopologicalSort()
        {
            ExecutionOrder.Clear();
            outgoing.Clear();
            // Node Id: No. of incoming connections
            Dictionary<int, int> indegree = new();
            // Node Id: [outgoing Node Id]
            Dictionary<int, List<int>> nodeMap = new();

            foreach (int id in Nodes.Keys)
            {
                indegree[id] = 0;
                nodeMap[id] = new List<int>();
                outgoing[id] = new List<Connection>();
            }

            // Maps how many incoming connections to each node
            foreach (Connection Conn in Connections)
            {
                int u = Conn.InNode.Id;
                int v = Conn.OutNode.Id;

                nodeMap[u].Add(v);
                indegree[v]++;
                if (Conn.Enabled)
                    outgoing[Conn.InNode.Id].Add(Conn);

            }

            Queue<int> queue = new();

            foreach (var kv in indegree)
                if (kv.Value == 0)
                    queue.Enqueue(kv.Key);

            while (queue.Count > 0)
            {
                int u = queue.Dequeue();
                ExecutionOrder.Add(u);

                foreach (int v in nodeMap[u])
                {
                    indegree[v]--;
                    if (indegree[v] == 0)
                        queue.Enqueue(v);
                }
            }
        }

        public double[] Forward(double[] inputs)
        {
            if (ExecutionOrder.Count == 0)
                TopologicalSort();

            // Reset all node inputs
            foreach (var node in Nodes.Values)
            {
                node.Input = 0.0;
            }

            // Initialise inputs
            int inputIndex = 0;
            foreach (var node in Nodes.Values)
            {
                if (node.Type == NodeType.Input)
                    node.Output = inputs[inputIndex++];

                if (node.Type == NodeType.Bias)
                    node.Output = 1.0;
            }

            // Forward Propagation
            foreach (int nodeID in ExecutionOrder)
            {
                Node node = Nodes[nodeID];

                if (node.Type != NodeType.Input && node.Type != NodeType.Bias)
                    node.Activate();

                foreach (Connection conn in outgoing[nodeID])
                {
                    if (conn.Enabled)
                        conn.OutNode.Input += node.Output * conn.Weight;
                }
            }
            // Collect outputs
            List<double> outputs = new();

            foreach (var node in Nodes.Values)
            {
                if (node.Type == NodeType.Output)
                    outputs.Add(node.Output);
            }

            return outputs.ToArray();
        }
        public void DebugPrint()
        {
            Debug.WriteLine($"Genome ID: {ID}");
            Debug.WriteLine($"Nodes ({Nodes.Count} total):");
            foreach (var node in Nodes.Values)
            {
                Debug.WriteLine($"  Id={node.Id}, Type={node.Type}, Layer={node.Layer}, Output={node.Output:F3}");
            }

            Debug.WriteLine($"Connections ({Connections.Count} total):");
            foreach (var conn in Connections)
            {
                Debug.WriteLine(
                    $"  {conn.InNode.Id} -> {conn.OutNode.Id}, Innovation={conn.InnovationNumber}, " +
                    $"Weight={conn.Weight:F3}, Enabled={conn.Enabled}"
                );
            }

            Debug.WriteLine("Execution Order: " + string.Join(",", ExecutionOrder));
            Debug.WriteLine("-----------");
        }
    }
}
