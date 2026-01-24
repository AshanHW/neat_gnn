using System;

namespace NEAT_GNN.Core
{
    public class Species
    {
        public int Id;
        public Genome Representative;
        public List<Genome> Members = new();

        public double TotalAdjustedFitness;
        public double AverageAdjustedFitness;
        public int OffspringCount;

        public Species(int id, Genome representative)
        {
            Id = id;
            Representative = representative;
            Members = new List<Genome> { representative };
        }
        public void ClearMembers()
        {
            Members.Clear();
            TotalAdjustedFitness = 0.0;
            AverageAdjustedFitness = 0.0;
            OffspringCount = 0;
        }
    }
}
