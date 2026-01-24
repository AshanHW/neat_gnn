using System;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NEAT_GNN.Core
{
    public class NodeData
    {
        public int Id { get; set; }
        public NodeType Type { get; set; }
    }
    public class ConnectionData
    {
        public int InNode { get; set; }
        public int OutNode { get; set; }
        public double Weight { get; set; }
        public bool Enabled { get; set; }
        public int InnovationNumber { get; set; }
    }
    public class GenomeData
    {
        public int ID { get; set; }
        public double RawFitness { get; set; }
        public List<NodeData> Nodes { get; set; }
        public List<ConnectionData> Connections { get; set; }
    }
    public class SingleGenomeSave
    {
        public int Generation { get; set; }
        public int GlobalGenomeId { get; set; }
        public int GlobalInnovationId { get; set; }
        public GenomeData Genome { get; set; }
    }
    public class InnovationTracker
    {
        public int currentInnovation = 0;
        public Dictionary<(int, int), int> connectionInnovations = new();
        public int GetInnovation(int inNodeId, int outNodeId)
        {
            var key = (inNodeId, outNodeId);

            // If already exists, return it
            if (connectionInnovations.TryGetValue(key, out int innov))
            {
                return innov;
            }

            // Else create new innovation number
            int newInnov = currentInnovation++;
            connectionInnovations[key] = newInnov;
            return newInnov;
        }
    }
    public class GenomeIdTracker
    {
        private int currentId = 0;
        public int GetNewId() => currentId++;
    }
    public class GenomeHandler
    {
        public int PopulationSize;
        public Dictionary<int, Genome> Genomes;
        public List<Species> SpeciesList;
        public InnovationTracker InnovationTracker;
        public GenomeIdTracker GenomeIdTracker;
        public Random Rng;

        public double C1; //excess
        public double C2; //disjoint
        public double C3; //weigh diff

        public double WeightMutationRate;
        public double AddConnectionRate;
        public double AddNodeRate;

        public GenomeHandler(int numOfgenomes, int numOfInputnodes, int numOfOutputnodes,
            WeightInitMode initMode, double[] compatibilityCoeffs, double[]MutationParams)
        {
            PopulationSize = numOfgenomes;
            Genomes = new();
            SpeciesList = new();
            InnovationTracker = new InnovationTracker();
            GenomeIdTracker = new GenomeIdTracker();
            Rng = new Random();
            C1 = compatibilityCoeffs[0];
            C2 = compatibilityCoeffs[1];
            C3 = compatibilityCoeffs[2];
            WeightMutationRate = MutationParams[0];
            AddConnectionRate = MutationParams[1];
            AddNodeRate = MutationParams[2];

            InitialisePopulation(numOfInputnodes, numOfOutputnodes, initMode);
        }
        public void InitialisePopulation(int numOfInputnodes, int numOfOutputnodes, WeightInitMode initMode)
        {
            // Create one genome
            // create nodes
            Dictionary<int, Node> inputNodes = new();
            Dictionary<int, Node> outputNodes = new();

            for (int i = 0; i < numOfInputnodes; i++)
            {
                inputNodes[i] = new Node(i, NodeType.Input, 0, ActivationFunctions.Linear);
            }
            // bias node
            inputNodes[numOfInputnodes] = new Node(numOfInputnodes, NodeType.Bias, 0,ActivationFunctions.Linear);

            for (int i = 0; i < numOfOutputnodes; i++)
            {
                int nodeId = numOfInputnodes + 1 + i;
                outputNodes[i] = new Node(nodeId, NodeType.Output, 1, ActivationFunctions.Sigmoid);
            }
            Dictionary<int, Node> allNodes = new Dictionary<int, Node>();
            foreach (var kv in inputNodes)
                allNodes[kv.Key] = kv.Value;
            foreach (var kv in outputNodes)
                allNodes[kv.Value.Id] = kv.Value;
            // create connections
            List<Connection> connections = new();
            foreach (var input in inputNodes)
            {
                foreach (var output in outputNodes)
                {
                    int innov = InnovationTracker.GetInnovation(input.Key, output.Key);
                    connections.Add(new Connection(input.Value, output.Value, innov, 0.0, true));
                }
            }
            // create the genome
            Genome ogGenome = new Genome(GenomeIdTracker.GetNewId(), allNodes, connections);
            Genomes.Add(0, ogGenome);

            //clone genomes
            for (int i = 1; i < PopulationSize; i++)
            {
                int newID = GenomeIdTracker.GetNewId();
                Genomes.Add(newID, CloneGenome(ogGenome, newID));
            }
            foreach (var kvp in Genomes)
            {
                Genome g = kvp.Value;
                g.InitialiseWeights(Rng, WeightInitMode.RandomUniform);
            }

        }
        private Genome CloneGenome(Genome originalGenome, int NewID)
        {
            var nodes = CloneNodes(originalGenome.Nodes);
            var connections = CloneConnections(originalGenome.Connections, nodes);

            return new Genome(NewID, nodes, connections);
        }
        private Dictionary<int, Node> CloneNodes(Dictionary<int, Node> original)
        {
            var clone = new Dictionary<int, Node>(original.Count);

            foreach (var kvp in original)
            {
                Node n = kvp.Value;

                clone[n.Id] = new Node(n.Id, n.Type, n.Layer, n.Activation);
            }
            return clone;
        }
        private List<Connection> CloneConnections(List<Connection> original, Dictionary<int, Node> clonedNodes)
        {
            var clone = new List<Connection>(original.Count);

            foreach (Connection c in original)
            {
                Node inNode = clonedNodes[c.InNode.Id];
                Node outNode = clonedNodes[c.OutNode.Id];

                clone.Add(new Connection(
                    inNode,
                    outNode,
                    c.InnovationNumber,
                    c.Weight,
                    c.Enabled
                ));
            }

            return clone;
        }
        public void Speciate(List<Genome> Population, double compThreshold)
        {
            // Step 0: clear previous members
            foreach (var species in SpeciesList)
                species.ClearMembers();

            foreach (var genome in Population)
            {
                bool assigned = false;

                foreach (var species in SpeciesList)
                {
                    double distance = CompatibilityDistance(genome, species.Representative);
                    if (distance < compThreshold)
                    {
                        species.Members.Add(genome);
                        assigned = true;
                        break;
                    }
                }

                if (!assigned)
                {
                    int newSpeciesId = SpeciesList.Count > 0 ? SpeciesList.Max(s => s.Id) + 1 : 0;
                    Species newSpecies = new Species(newSpeciesId, genome);
                    SpeciesList.Add(newSpecies);
                }
            }
        }
        public void AdjustFitness()
        {
            foreach (var species in SpeciesList)
            {
                int speciesSize = species.Members.Count;

                foreach (var genome in species.Members)
                {
                    // Adjusted fitness = raw fitness / species size
                    genome.AdjustedFitness = genome.RawFitness / speciesSize;
                }

                // Compute species totals
                species.TotalAdjustedFitness = species.Members.Sum(g => g.AdjustedFitness);
                species.AverageAdjustedFitness = species.TotalAdjustedFitness / species.Members.Count;
            }
        }
        public void SelectNewRepresentative()
        {
            Random rng = new Random();
            foreach (var species in SpeciesList)
            {
                if (species.Members.Count > 0)
                {
                    int index = rng.Next(species.Members.Count);
                    species.Representative = species.Members[index];
                }
                else
                {
                    // Species extinct
                    species.Representative = null;
                    SpeciesList.Remove(species);
                }
            }
        }
        public void ComputeOffspring()
        {
            double totalAverageFitness = SpeciesList.Sum(s => s.AverageAdjustedFitness);

            foreach (var species in SpeciesList)
            {
                // Proportional allocation
                species.OffspringCount = (int)Math.Round(
                    (species.AverageAdjustedFitness / totalAverageFitness) * PopulationSize
                );
            }

            // Adjust rounding to ensure total offspring = population size
            int assigned = SpeciesList.Sum(s => s.OffspringCount);
            int diff = PopulationSize - assigned;

            // Add/subtract from largest species
            if (diff != 0)
            {
                Species largest = SpeciesList.OrderByDescending(s => s.Members.Count).First();
                largest.OffspringCount += diff;
            }
        }
        public void SetFitness(Dictionary<int, double> fitnessScores)
        {
            foreach (var kvp in fitnessScores)
                Genomes[kvp.Key].RawFitness = kvp.Value;
        }
        public double CompatibilityDistance(Genome g1, Genome g2)
        {
            // Sort connections by innovation number
            var genes1 = g1.Connections.OrderBy(c => c.InnovationNumber).ToList();
            var genes2 = g2.Connections.OrderBy(c => c.InnovationNumber).ToList();

            int i = 0, j = 0;
            int excess = 0;
            int disjoint = 0;
            double weightDiffSum = 0.0;
            int matching = 0;

            int maxInnov1 = genes1.Last().InnovationNumber;
            int maxInnov2 = genes2.Last().InnovationNumber;

            while (i < genes1.Count && j < genes2.Count)
            {
                int innov1 = genes1[i].InnovationNumber;
                int innov2 = genes2[j].InnovationNumber;

                if (innov1 == innov2)
                {
                    matching++;
                    weightDiffSum += Math.Abs(genes1[i].Weight - genes2[j].Weight);
                    i++;
                    j++;
                }
                else if (innov1 < innov2)
                {
                    disjoint++;
                    i++;
                }
                else
                {
                    disjoint++;
                    j++;
                }
            }

            // Remaining genes are excess
            excess += (genes1.Count - i);
            excess += (genes2.Count - j);

            int N = Math.Max(genes1.Count, genes2.Count);
            if (N < 20) N = 1; // NEAT paper convention

            double avgWeightDiff = matching > 0 ? weightDiffSum / matching : 0.0;

            return (C1 * excess) / N + (C2 * disjoint) / N + (C3 * avgWeightDiff);
        }
        private List<Genome> SelectParents(Species species, double survivalRate)
        {
            int survivorCount = Math.Max(1, (int)(species.Members.Count * survivalRate));

            return species.Members.OrderByDescending(g => g.RawFitness).Take(survivorCount).ToList();
        }
        public Genome Crossover(Genome parent1, Genome parent2, int childId)
        {
            // Ensure parent1 is more fit
            if (parent2.RawFitness > parent1.RawFitness)
                (parent1, parent2) = (parent2, parent1);

            var childNodes = CloneNodes(parent1.Nodes);
            var childConnections = new List<Connection>();

            var genes1 = parent1.Connections.ToDictionary(c => c.InnovationNumber);
            var genes2 = parent2.Connections.ToDictionary(c => c.InnovationNumber);

            foreach (var kvp in genes1)
            {
                int innov = kvp.Key;
                Connection gene1 = kvp.Value;

                if (genes2.TryGetValue(innov, out Connection gene2))
                {
                    // Matching gene
                    Connection chosen = Rng.NextDouble() < 0.5 ? gene1 : gene2;

                    bool enabled = true;
                    if (!gene1.Enabled || !gene2.Enabled)
                        enabled = Rng.NextDouble() > 0.75;

                    childConnections.Add(new Connection(
                        childNodes[chosen.InNode.Id],
                        childNodes[chosen.OutNode.Id],
                        innov,
                        chosen.Weight,
                        enabled
                    ));
                }
                else
                {
                    // Disjoint or excess → from fitter parent only
                    childConnections.Add(new Connection(
                        childNodes[gene1.InNode.Id],
                        childNodes[gene1.OutNode.Id],
                        gene1.InnovationNumber,
                        gene1.Weight,
                        gene1.Enabled
                    ));
                }
            }

            Genome child = new Genome(childId, childNodes, childConnections);
            return child;
        }
        public void MutateWeights(Genome g)
        {
            foreach (var c in g.Connections)
            {
                if (Rng.NextDouble() < 0.9)
                    c.Weight += Rng.NextDouble() * 0.2 - 0.1;
                else
                    c.Weight = Rng.NextDouble() * 2 - 1;
            }
        }
        public void MutateAddConnection(Genome g)
        {
            var possiblePairs =
                from a in g.Nodes.Values
                from b in g.Nodes.Values
                where a.Layer < b.Layer                     // feedforward only
                where a.Type != NodeType.Output              // no outgoing from output
                where b.Type != NodeType.Input               // no incoming to input
                where b.Type != NodeType.Bias                // no incoming to bias
                select (a, b);

            var pair = possiblePairs.OrderBy(_ => Rng.Next()).FirstOrDefault(p => !g.Connections.Any(c =>
                        c.InNode.Id == p.a.Id &&
                        c.OutNode.Id == p.b.Id));

            if (pair == default) return;

            int innov = InnovationTracker.GetInnovation(pair.a.Id, pair.b.Id);

            g.Connections.Add(new Connection(
                pair.a,
                pair.b,
                innov,
                Rng.NextDouble() * 2 - 1,
                true
            ));
        }
        public void MutateAddNode(Genome g)
        {
            var conn = g.Connections
                .Where(c => c.Enabled)
                .OrderBy(_ => Rng.Next())
                .FirstOrDefault();

            if (conn == null) return;

            conn.Enabled = false;

            int inLayer = conn.InNode.Layer;
            int outLayer = conn.OutNode.Layer;
            int newLayer = inLayer + 1;

            // Shift layers if collision
            if (outLayer == newLayer)
            {
                foreach (var node in g.Nodes.Values)
                {
                    if (node.Layer >= newLayer)
                        node.Layer++;
                }
            }

            int newNodeId = g.Nodes.Keys.Max() + 1;
            Node newNode = new Node(newNodeId, NodeType.Hidden, newLayer, ActivationFunctions.Tanh);
            g.Nodes[newNodeId] = newNode;

            int innov1 = InnovationTracker.GetInnovation(conn.InNode.Id, newNodeId);
            int innov2 = InnovationTracker.GetInnovation(newNodeId, conn.OutNode.Id);

            g.Connections.Add(new Connection(
                conn.InNode,
                newNode,
                innov1,
                1.0,
                true
            ));

            g.Connections.Add(new Connection(
                newNode,
                conn.OutNode,
                innov2,
                conn.Weight,
                true
            ));
        }
        public Dictionary<int, Genome> Reproduce()
        {
            Dictionary<int, Genome> nextGen = new();

            foreach (var species in SpeciesList)
            {
                var parents = SelectParents(species, 0.2);

                // Elitism: keep best genome
                Genome elite = parents[0];
                int eliteId = GenomeIdTracker.GetNewId();
                nextGen[eliteId] = CloneGenome(elite, eliteId);

                for (int i = 1; i < species.OffspringCount; i++)
                {
                    Genome child;

                    if (parents.Count > 1)
                    {
                        Genome p1 = parents[Rng.Next(parents.Count)];
                        Genome p2 = parents[Rng.Next(parents.Count)];
                        int childId = GenomeIdTracker.GetNewId();
                        child = Crossover(p1, p2, childId);
                    }
                    else
                    {
                        int childId = GenomeIdTracker.GetNewId();
                        child = CloneGenome(parents[0], childId);
                    }

                    if (Rng.NextDouble() < WeightMutationRate)
                        MutateWeights(child);
                    if (Rng.NextDouble() < AddConnectionRate)
                        MutateAddConnection(child);
                    if (Rng.NextDouble() < AddNodeRate)
                        MutateAddNode(child);

                    nextGen[child.ID] = child;
                }
            }

            return nextGen;
        }
        public double[][] Forward(double[][] inputBatch)
        {
            double[][] outputBatch = new double[inputBatch.Length][];
            Parallel.For(0, inputBatch.Length, i =>
            {
                outputBatch[i] = Genomes[i].Forward(inputBatch[i]);
            });
            return outputBatch;
        }
        public void EvolveOneGeneration(Dictionary<int, double> fitnessScores, double compatibilityThreshold)
        {
            // 1. Assign fitness
            SetFitness(fitnessScores);

            // 2. Speciation
            Speciate(Genomes.Values.ToList(), compatibilityThreshold);

            // 3. Fitness sharing
            AdjustFitness();

            // 4. Representative selection
            SelectNewRepresentative();

            // 5. Offspring allocation
            ComputeOffspring();

            // 6. Reproduction
            Genomes = Reproduce();
        }
        public void SaveGenome(Genome genome, int generation, string savePath)
        {
            var genomeData = new GenomeData
            {
                ID = genome.ID,
                RawFitness = genome.RawFitness,
                Nodes = genome.Nodes.Values.Select(n => new NodeData
                {
                    Id = n.Id,
                    Type = n.Type
                }).ToList(),
                Connections = genome.Connections.Select(c => new ConnectionData
                {
                    InNode = c.InNode.Id,
                    OutNode = c.OutNode.Id,
                    Weight = c.Weight,
                    Enabled = c.Enabled,
                    InnovationNumber = c.InnovationNumber
                }).ToList()
            };

            var saveObject = new SingleGenomeSave
            {
                Generation = generation,
                Genome = genomeData,
                GlobalGenomeId = genome.ID,
                GlobalInnovationId = InnovationTracker.currentInnovation
            };

            string json = JsonSerializer.Serialize(saveObject, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(savePath, json);
        }
        public void LoadWeights(int genomeID, string savePath)
        {
            if (!Genomes.ContainsKey(genomeID))
                throw new Exception($"Genome {genomeID} not found in handler.");

            string json = File.ReadAllText(savePath);
            var saved = JsonSerializer.Deserialize<SingleGenomeSave>(json);

            Genome genome = Genomes[genomeID];

            foreach (var connData in saved.Genome.Connections)
            {
                var conn = genome.Connections.FirstOrDefault(c => c.InnovationNumber == connData.InnovationNumber);
                if (conn != null)
                {
                    conn.Weight = connData.Weight;
                    conn.Enabled = connData.Enabled;
                }
                else
                {
                    Console.WriteLine($"Warning: Connection {connData.InnovationNumber} not found in genome {genomeID}");
                }
            }

            genome.RawFitness = saved.Genome.RawFitness;
        }
        public void LoadWeightsBatch(Dictionary<int, string> genomePaths)
        {
            foreach (var kvp in genomePaths)
            {
                LoadWeights(kvp.Key, kvp.Value);
            }
        }
    }
}
