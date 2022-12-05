@testset "test runexperiments" begin
    using GenerativeRecoveries: runexperimenttensor
    using DataFrames: DataFrame
    experimentfn(firstsetup, secondsetup, fixed) = ((:result => firstsetup * secondsetup * fixed),)
    experimentsetup = ([1, 3, 9], [2, 4])
    experimentlabels = (:firstsetup, :secondsetup)
    @test runexperimenttensor(experimentfn, experimentsetup, experimentlabels, 2) isa DataFrame
end