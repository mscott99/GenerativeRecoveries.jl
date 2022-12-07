# dataframes to replace runexperiments
using Flux
using BSON: @load
using Revise
using FFTW: plan_dct
using GenerativeRecoveries
using GenerativeRecoveries: wrap_model_withreshape, plot_recovery_errors_tocompare_frequencysamplingalgorithms, _setupMNISTimagesignals, samplefrequenciesuniformly, samplefrequenciesuniformlyanddeterministically, _setupmodel, IndexedMatrix, recoversignal, runexperimenttensor, _setupfrequencies, _plot_tables_ofrecoveries, plot_MNISTrecoveries, plot_MNISTrecoveryerrors
@load "savedmodels/more_incoherentepoch20" model
model = wrap_model_withreshape(model)
using DataFrames
using DataFrames: allcombinations, DataFrame, transform!
using LinearAlgebra: norm
using Statistics
presigmoid = true
inrange = true
kwargs = (;)

img_size = (28, 28)
numbers = [1,1, 2,2, 3,3,4,4,5,5,6,6,7,7,8,8,9,9]
aimed_ms = [2, 4, 8, 16]
set_aog_theme!()
samplingfunctions = [samplefrequenciesuniformly, samplefrequenciesuniformlyanddeterministically]
using Test 

re = plot_recovery_errors_tocompare_frequencysamplingalgorithms(model, numbers, aimed_ms, samplefrequenciesuniformly, samplefrequenciesuniformlyanddeterministically, img_size; max_iter = 1000)
save("./experiment_data/compare_frequency_samplings.svg",re)

presigmoid = true
inrange = true
sampleevenlyfn
model = _setupmodel(model; presigmoid)
    samplingfnlabels = ["even", "uneven"]
    samplingfndict = Dict("even" => sampleevenlyfn, "uneven" => sampleunevenlyfn)
    decoder = model.decoder
    truesignals = _setupMNISTimagesignals(images, model; presigmoid, inrange, datasplit, kwargs...)
    # not well adapted to the in range case with many models; for that we need a notion of dependence when running experiments. Probably a callback.
    pdct = plan_dct(truesignals[1])

    frame = allcombinations(DataFrame, (:truesignal => truesignals), (:numfrequencies => aimed_ms), (:algname => samplingfnlabels))
    transform!(frame, [:numfrequencies, :algname] => ByRow((m, algname) -> samplingfndict[algname](m, img_size)) => :sampledfrequencies)
    transform!(frame, [:truesignal, :sampledfrequencies] => ByRow((truesignal, frequency) -> experimentgetrelative_error(truesignal, frequency, decoder, pdct, recoversignal; kwargs...)) => AsTable)

    frame = combine(groupby(frame, [:numfrequencies, :algname]), :relerr, :algname, :relerr => mean => :meanrelerr, :relerr => std => :std_deviation)
    transform!(frame, [:meanrelerr, :std_deviation] => ByRow((meanrelerr, std_deviation) -> meanrelerr .+ std_deviation) => :topuncert)
    transform!(frame, [:meanrelerr, :std_deviation] => ByRow((meanrelerr, std_deviation) -> meanrelerr .- std_deviation) => :botuncert)

    mapmargins = mapping(:numfrequencies, :botuncert, :topuncert) * visual(AlgebraOfGraphics.Band) * visual(alpha=0.2)
    mapmeans = mapping(:numfrequencies, :meanrelerr) * visual(AlgebraOfGraphics.Lines)
    mapscatter = mapping(:numfrequencies, :relerr) * visual(AlgebraOfGraphics.Scatter)
    plt = AlgebraOfGraphics.data(frame) * (mapmargins + mapmeans + mapscatter) * mapping(color=:algname => "Sampling Type")
    draw(plt, axis=(; xscale=log10, yscale=log10))

statsdf = combine(groupby(df, [:numfrequencies, :algname]), :relerr, :algname, :relerr => mean => :meanrelerr, :relerr => std => :std_deviation)
transform!(statsdf, [:meanrelerr, :std_deviation] => ByRow((meanrelerr, std_deviation) -> meanrelerr .+ std_deviation) => :topuncert)
transform!(statsdf, [:meanrelerr, :std_deviation] => ByRow((meanrelerr, std_deviation) -> meanrelerr .- std_deviation) => :botuncert)

using CairoMakie

mapmargins = mapping(:numfrequencies, :botuncert, :topuncert)*visual(AlgebraOfGraphics.Band)*visual(alpha=0.2)
mapmeans = mapping(:numfrequencies, :meanrelerr)*visual(AlgebraOfGraphics.Lines)
mapscatter = mapping(:numfrequencies, :relerr)*visual(AlgebraOfGraphics.Scatter)
plt = data(statsdf)*(mapmargins + mapmeans + mapscatter)*mapping(color=:algname => "Sampling Type")
draw(plt, axis = (;xscale=log10, yscale=log10))



#eachstats = groupby(statsdf, :frequencysamplingalgorithm)
using AlgebraOfGraphics: Band

using AlgebraOfGraphics: Lines, linesfill
scatterplt = data(df) * mapping(:numfrequencies, :relerr)
linesplt = data(statsdf) * mapping(:numfrequencies, :meanrelerr) *mapping(color=:algname)* |> draw

plt = (data(df) * mapping(:numfrequencies, :relerr) + ) * mapping(color=:algname) |> draw
axis = (width=225, height=225)


draw(plt; axis)

data(statsdf) * mapping(:numfrequencies => "Number of frequencies", :meanrelerr => "Mean relative error") * mapping(color=:algorithmname) |> draw



#stats = select(groupby(df, :numfrequencies), :relerr => mean => :meanrelerr, :relerr => std => :std_deviation)

scatter!(axes, df[!, :numfrequencies], df[!, :relerr])
fig
@show frame[1, :sampledfrequencies]



transform!(frame, [:truesignal, :numfrequencies] => (truesignal, numfrequencies) -> truesignal .* numfrequencies)
freqs = [frequencyfunction(aimed_m) for aimed_m in aimedmeasurementnumbers for frequencyfunction in samplingfunctions]
# make table of frequencies and specify two labels. pass this to runexperiments. 
_setupfrequencies(aimedmeasurementnumbers, size(truesignals[1]))



setuplabels = (:truesignal, :frequency)
experimentsetup = (truesignals, freqs)
fixedsetup = (decoder, pdct, recoveryfn)
resultdataframe = runexperimenttensor(experimentfn, experimentsetup, setuplabels, fixedsetup...; kwargs...)
return resultdataframe
plot_recovery_errors(resultdataframe)


# Testing plotMNIST
using GenerativeRecoveries: plot_MNISTrecoveries, wrap_model_withreshape, FullVae, VaeEncoder, logrange
using Flux
using BSON: @load
using Revise
@load "savedmodels/more_incoherentepoch20" model
model = wrap_model_withreshape(model)
plot_MNISTrecoveryerrors(model, [16, 32], [2, 3], max_iter=10)

frequencyfunction
freqs = [frequencyfunction(aimed_m) for aimed_m in aimedmeasurementnumbers for frequencyfunction in samplingfunctions]

using DataFrames: DataFrame, groupby, convert
using Test
using GenerativeRecoveries: runexperimenttensor
using DataFrames: DataFrame
experimentfn(; firstsetup, secondsetup, fixed) = (; :result => firstsetup * secondsetup * fixed)
experimentsetup = Dict(:firstsetup => [1, 3, 9], :secondsetup => [2, 4])
fixedsetup = Dict(:fixed => 2)

a = DataFrame(a=[1, 2, 3, 4], b=[1.9, 2.0, 4.5, 6.5])

@test runexperimenttensor(experimentfn, experimentsetup, fixedsetup) isa DataFrame
secondmodel = model
@load "savedmodels/bounded_morecoherencematchingepoch20" model
model = wrap_model_withreshape(model)
plot_MNISTrecoveries([model, secondmodel], [16, 32, 64], [2, 3, 8, 9], max_iter=10, multithread=false)

# testexperiments Test runexperiment tensor ----------------
using Test
using Flux
using BSON: @load
using GenerativeRecoveries: FullVae, recoversignal, wrap_model_withreshape, VaeEncoder, _setupmodels, _setupMNISTimagesignals, _setupsampledfrequencies, IndexedMatrix, runexperimenttensor
using FFTW: plan_dct
using LinearAlgebra: norm

@load "savedmodels/bounded_morecoherencematchingepoch20" model
secondmodel = wrap_model_withreshape(model)

@load "savedmodels/more_incoherentepoch20" model
model = wrap_model_withreshape(model)
@test secondmodel isa FullVae

model = _setupmodels(secondmodel)[1]
images = [2, 3, 5]
aimedmeasurementnumbers = [22, 33, 45]
truesignals = _setupMNISTimagesignals(images, secondmodel)
freqs = _setupsampledfrequencies(aimedmeasurementnumbers, size(truesignals[1]))
pdct = plan_dct(truesignals[1])
pdct * truesignals[1]
#FatFFTPlan(pdct, frequencies[1])
# mystery error here
freqs = _setupsampledfrequencies([16, 32, 64], size(truesignals[1]))

# As = [IndexedMatrix(ParallelMatrix(pdct), freq) for freq in freqs]
#As = map(x -> FatFFTPlan(pdct, x), frequencies)

experimentsetup = (truesignals, freqs)

recoveryfn = recoversignal

function experimentfn(; truesignal, frequency, pdct, model, recoveryfn, kwargs...) # pass frequencies only
    A = IndexedMatrix(deepcopy(pdct), frequency)
    measurements = A * truesignal
    recoveryimg = recoveryfn(measurements, A, model, max_iter=10; kwargs...)
    relativeerr = norm(recoveryimg .- truesignal) / norm(truesignal)
    (recoveryimg, relativeerr)
end

array_in = (experimentsetup[i][1] for i in 1:length(experimentsetup))

decoder = model.decoder
# As[1] * truesignals[1]

typeof(truesignals[1])
size(truesignals[1])

experiment_result = experimentfn(array_in..., pdct, decoder, recoveryfn, max_iter=10)

@test runexperimenttensor(experimentfn, ([freqs[1]],), images[1], pdct, decoder, recoveryfn) isa Vector
@test runexperimenttensor(experimentfn, experimentsetup, pdct, decoder, recoveryfn, max_iter=10, multithread=true) isa Matrix


# struct ParallelFFTPlan{T<:AbstractFFTs.Plan}
#     pdct::AbstractArray{T}
# end

# ParallelFFTPlan(pdct::AbstractFFTs.Plan) = ParallelFFTPlan([deepcopy(pdct) for i in 1:nthreads()])

# *(a::ParallelFFTPlan, x::AbstractArray) = a.pdct[threadid()] * x

# struct FatFFTPlan{T<:Union{AbstractFFTs.Plan,ParallelFFTPlan},F<:AbstractArray{Bool}}
#     p::T
#     freqs::F
# end

# *(A::FatFFTPlan, x::AbstractArray) = (A.p*x)[A.freqs]

# struct IndexedMatrix{T,L}
#     A::T
#     indices::L
# end

# import Base: *

# *(indexedMatrix::IndexedMatrix, x::AbstractArray) = (indexedMatrix.A*x)[indexedMatrix.indices]

# using Base.Threads

# indices = reshape(collect(CartesianIndices(length.(experimentsetup))), :)

# decoder = secondmodel.decoder

# @threads for index in indices
#     results[index] = experimentfn((experimentsetup[i][index[i]] for i in 1:length(experimentsetup))..., decoder, recoveryfn, max_iter=3000)
# end

# results


# using Colors: Gray
# using GenerativeRecoveries: logrange
# @time plot_MNISTrecoveries(wrapped_model, logrange(16, 783, 6), [1, 2, 3, 4, 5, 6], max_iter=3000, multithread=false)

# @test plot_MNISTrecoveries([wrapped_model, secondmodel], [2, 4], [3, 5], max_iter=10) isa AbstractArray{<:Figure,1}

# @test plot_MNISTrecoveries(wrapped_model, [2, 4], [3, 5], relaxed_recover, optimlayers=[2], max_iter=10) isa Figure
# @test plot_MNISTrecoveries(wrapped_model, [2, 4], [3, 5], [relaxed_recover, recoversignal], optimlayers=[2], max_iter=10) isa AbstractArray{<:Figure}
# using Base.Threads

# indices = reshape(collect(CartesianIndices(length.(experimentsetup))), :)

# decoder = secondmodel.decoder

# @threads for index in indices
#     results[index] = experimentfn((experimentsetup[i][index[i]] for i in 1:length(experimentsetup))..., decoder, recoveryfn, max_iter=3000)
# end

# results


# using Colors: Gray
# using GenerativeRecoveries: logrange
# @time plot_MNISTrecoveries(wrapped_model, logrange(16, 783, 6), [1, 2, 3, 4, 5, 6], max_iter=3000, multithread=false)

# @test plot_MNISTrecoveries([wrapped_model, secondmodel], [2, 4], [3, 5], max_iter=10) isa AbstractArray{<:Figure,1}

# @test plot_MNISTrecoveries(wrapped_model, [2, 4], [3, 5], relaxed_recover, optimlayers=[2], max_iter=10) isa Figure
# @test plot_MNISTrecoveries(wrapped_model, [2, 4], [3, 5], [relaxed_recover, recoversignal], optimlayers=[2], max_iter=10) isa AbstractArray{<:Figure}




