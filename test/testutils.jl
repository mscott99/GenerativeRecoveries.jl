@testset "test runexperiments" begin
    using GenerativeRecoveries: runexperimenttensor
    using DataFrames: DataFrame
    experimentfn(firstsetup, secondsetup, fixed) = ((:result => firstsetup * secondsetup * fixed),)
    experimentsetup = ([1, 3, 9], [2, 4])
    experimentlabels = (:firstsetup, :secondsetup)
    @test runexperimenttensor(experimentfn, experimentsetup, experimentlabels, 2) isa DataFrame
end

@testset "test sampling functions" begin
    using GenerativeRecoveries: samplefromarray, samplesmallestfrequencies, samplefrequenciesuniformlyanddeterministically, _modminimizeindex
    @test _modminimizeindex(CartesianIndex(1, 28), (28, 28)) == [0, 1]
    @test (sum(samplesmallestfrequencies(1000, (20, 25, 30, 20, 10), dct=false)) / 1000.0 - 1.0) ≤ 0.1
    @test (sum(samplesmallestfrequencies(1000, (20, 25, 30, 20, 10), dct=true)) / 1000.0 - 1.0) ≤ 0.1
    testfreq = samplefrequenciesuniformlyanddeterministically(44, 26, (60, 40, 2, 1, 1))
    @test sum(testfreq) == 44 + 26
    @test length(samplefromarray([1, 2, 3], 2)) == 2
end

@testset "test image setup" begin
    using GenerativeRecoveries: setupCELEBAimagesignals
    allimages = setupCELEBAimagesignals(3)
    @test length(allimages) == 3
    @test allimages[1] isa Array{Float32,3}
end

