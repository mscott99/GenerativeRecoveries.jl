@testset "test runexperiments" begin
    using GenerativeRecoveries: runexperimenttensor
    using DataFrames: DataFrame
    experimentfn(firstsetup, secondsetup, fixed) = ((:result => firstsetup * secondsetup * fixed),)
    experimentsetup = ([1, 3, 9], [2, 4])
    experimentlabels = (:firstsetup, :secondsetup)
    @test runexperimenttensor(experimentfn, experimentsetup, experimentlabels, 2) isa DataFrame
end

@testset "test sampling functions" begin
    using GenerativeRecoveries: sampledeterministicallyfirstfrequencies, getdeterministicanduniformfrequencies
    @test sum(sampledeterministicallyfirstfrequencies(10000, (20, 25, 30, 20, 10))) / 10000 ≤ 0.1
    testfreq = getdeterministicanduniformfrequencies(44, 26, (60, 40, 2, 1, 1))
    @test sum(testfreq) == 44 + 26
end