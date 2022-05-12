using ReservoirComputing, Random, Plots, LinearAlgebra
using DelimitedFiles
using Downloads

data_path = Downloads.download("https://mantas.info/wp/wp-content/uploads/simple_esn/MackeyGlass_t17.txt", 
    string(pwd(),"/MackeyGlass_t17.txt"))
data = CuArray(reduce(hcat, convert(Matrix{Float32}, readdlm(data_path, ','))))

washout      = 100
train_len    = 2000
predict_len  = 2000

input_data   = data[:, 1:train_len]
target_data  = data[:, washout+2:train_len+1]
test_data    = data[:, train_len+2:train_len+predict_len+1]

Random.seed!(42)

esn = ESN(input_data; 
    variation = Default(),
    reservoir = RandSparseReservoir(1000, 1.25, 1.0),
    input_layer = WeightedLayer(1.0),
    reservoir_driver = RNN(leaky_coefficient=0.3),
    nla_type = NLADefault(),
    states_type = PaddedExtendedStates(),
    washout=washout)

training_method = StandardRidge(1e-8) 
output_layer = train(esn, target_data, training_method)    
output = esn(Generative(predict_len), output_layer)


println(sum(abs2.(test_data[:,1:500] .- output[:,1:500]))/500)

plot([test_data' output'], label = ["actual" "predicted"], 
    plot_title="Makey Glass Time Series",
    titlefontsize=20,
    legendfontsize=12,
    linewidth=2.5,
    xtickfontsize = 12,
    ytickfontsize = 12,
    size=(1080, 720))
savefig("mackeyglass_basic.png")
