using ReservoirComputing, Random, Plots, StatsBase
using DelimitedFiles

data = reduce(hcat, readdlm("santafe_laser.txt"))

train_len   = 5000
predict_len = 2000

training_input  = data[:, 1:train_len]
training_target = data[:, 2:train_len+1]
testing_input   = data[:,train_len+1:train_len+predict_len]
testing_target  = data[:,train_len+2:train_len+predict_len+1]

res_size = 300
res_radius = 1.4

#gru_res1 = RandSparseReservoir(res_size)
#gru_res2 = RandSparseReservoir(res_size)

#build ESN struct
Random.seed!(42)
esn = ESN(training_input; 
    reservoir = RandSparseReservoir(res_size, radius=res_radius),
    #reservoir_driver = GRU(reservoir=[gru_res1, gru_res2]),
    reservoir_driver = GRU())

#define training method
training_method = StandardRidge(0.0)

#obtain output layer
output_layer = train(esn, training_target, training_method)
output = esn(Predictive(testing_input), output_layer)

#comparison with ESN
esn_rnn = ESN(training_input; 
    reservoir = RandSparseReservoir(res_size, radius=res_radius),
    reservoir_driver = RNN())

output_layer    = train(esn_rnn, training_target, training_method)
output_rnn      = esn_rnn(Predictive(testing_input), output_layer)

println(msd(testing_target, output))
println(msd(testing_target, output_rnn))

plot([testing_target' output'], label=["actual" "predicted"], 
    plot_title="Santa Fe Laser",
    titlefontsize=20,
    legendfontsize=12,
    linewidth=2.5,
    xtickfontsize = 12,
    ytickfontsize = 12,
    size=(1080, 720))
savefig("gru.png")
