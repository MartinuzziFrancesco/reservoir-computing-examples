using ReservoirComputing, Plots, DelimitedFiles, LinearAlgebra, StatsBase
using DynamicalSystems
#=
data = reduce(hcat, readdlm("santafe_laser.txt"))

train_len = 7000
predict_len = 2000
=#
function split_data(data, train_len, predict_len; shift=1)
    training_input = data[:, shift:shift+train_len-1]
    training_target = data[:, shift+1:shift+train_len]
    testing_input = data[:,shift+train_len:shift+train_len+predict_len-1]
    testing_target = data[:,shift+train_len+1:shift+train_len+predict_len]

    training_input, training_target, testing_input, testing_target
end


ds = Systems.henon()
traj = trajectory(ds, 7000)
data = Matrix(traj)'
data = (data .-0.5) .* 2
shift = 200

train_len = 3000
predict_len = 2000

function NMSE(target, output)
   num = 0.0
   den = 0.0
   sums = []
   for i=1:size(target, 1)
       append!(sums, sum(target[i,:]))
   end
   for i=1:size(target, 2)
       num += norm(output[:,i]-target[:,i])^2.0
       den += norm(target[:,i]-sums./size(target, 2))^2.0
   end
   nmse = (num/size(target, 2))/(den/size(target, 2))
   return nmse
end


training_input, training_target, testing_input, testing_target = split_data(data, 
    train_len, predict_len)

#define input layer
input_layer = [MinimumLayer(0.85, IrrationalSample()), MinimumLayer(0.95, IrrationalSample())]

 #define reservoirs
reservoirs = [SimpleCycleReservoir(300, 0.7), 
     CycleJumpsReservoir(300, cycle_weight=0.7, jump_weight=0.2, jump_size=5)]

for i=1:length(reservoirs)
    esn = ESN(training_input;
        input_init = input_layer[i],
        reservoir_init = reservoirs[i])
    wout = train(esn, training_target, StandardRidge(0.001))
    output = esn(Predictive(testing_input), wout)
    #println(NMSE(testing_target, output))
    println(msd(testing_target, output))
    #plot(testing_target')
    #plot!(output')
    #savefig("test$i.png")
end


