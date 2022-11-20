using Plots: plot
using Colors
using MLDatasets: MNIST
using Flux
using BSON
using BSON: @load, @save
using Revise
using GenerativeRecoveries: addreshape_tomodel, FullVae
using GenerativeRecoveries

@load "savedmodels/more_incoherentepoch20" model

model = BSON.parse("savedmodels/incoherentepoch20")[:model]
typeof(model)
