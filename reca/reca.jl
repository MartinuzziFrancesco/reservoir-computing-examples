using ReservoirComputing, CellularAutomata, DelimitedFiles

input = readdlm("./5bitinput.txt", ',', Bool)
output = readdlm("./5bitoutput.txt", ',', Bool)

reca = RECA(input, DCA(90); 
    generations = 16,
    input_encoding = RandomMapping(16, 40))

output_layer = train(reca, output, StandardRidge(0.00001))
prediction = reca(Predictive(input), output_layer)
final_pred = convert(AbstractArray{Bool}, prediction .> 0.5)

final_pred == output
