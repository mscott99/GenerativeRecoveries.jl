function trainincoherentVae(vaelossfn, β, λ, α, F, model, pars::Flux.Params, traindata, opt::Flux.Optimise.AbstractOptimiser, numepochs, savedir, tblogdir; loginterval=10, label="", logginglossterms=false)
    # The training loop for the model
    tblogger = TBLogger(tblogdir)

    function klfromgaussian(μ, logvar)
        0.5 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0))
    end

    function l2reg(pars)
        sum(x -> sum(abs2, x), pars)
    end

    lastlayer = params(model.decoder[end])[1]
    #lastlayercoherence() = maximum(sqrt.(sum((F * lastlayer) .* (F * lastlayer), dims=2))) + norm(lastlayer' * lastlayer - I(500), 2)^2
    lastlayercoherence(lastlayer) = sqrt(maximum(sum(abs2, F * lastlayer, dims=2))) + sum(abs2, lastlayer' * lastlayer - I)


    #numbatches = length(data)
    @progress for epochnum in 1:numepochs
        for (step, x) in enumerate(traindata)

            loss, back = pullback(pars) do
                vaelossfn(x, F, model, pars, lastlayer, β, λ, α)
            end
            gradients = back(1.0f0)
            Flux.Optimise.update!(opt, pars, gradients)

            if step % loginterval == 0
                with_logger(tblogger) do

                    if logginglossterms
                        vaelossfn(x, F, model, pars, lastlayer, β, λ, α, tblogger)
                    end
                    @info "loss" epochnum loss
                end
            end
        end
        @save string(savedir, label, "epoch", epochnum) model opt epochnum β λ α label
    end

    @info "training complete!"
end

function train_incoherentVAE_onMNIST(; vaelossfn=VAEloss_boundedcoherence, numepochs=20, β=1.0f0, λ=1.0f-2, α=1.0f4, kwargs...)
    model = makeMNISTVae(512, 512, 16)
    batchsize = 64

    traindata = reshape(MNIST(Float32, :train).features[:, :, 1:end], 28^2, :)
    trainloader = DataLoader(traindata, batchsize=batchsize)

    F = dct(diagm(ones(28^2)), 2)

    trainincoherentVae(vaelossfn, β, λ, α, F, model, params(model), trainloader, Adam(), numepochs, "./reusefiles/models/", "./reusefiles/logs/"; kwargs...)
end

function VAEloss_unitarycoherence(x, F, model::FullVae, pars, lastlayer, β, λ, α)
    function klfromgaussian(μ, logvar)
        0.5 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0))
    end

    function l2reg(pars)
        sum(x -> sum(abs2, x), pars)
    end
    intermediate = model.encoder.encoderbody(x)
    μ = model.encoder.splitedμ(intermediate)
    logvar = model.encoder.splitedlogvar(intermediate)
    z = μ .+ randn(Float32, size(logvar)) .* exp.(0.5f0 .* logvar)
    x̂ = model.decoder(z)

    lastlayercoherence(lastlayer) = sqrt(maximum(sum(abs2, F * lastlayer, dims=2))) + sum(abs2, lastlayer' * lastlayer - I)

    logitbinarycrossentropy(x̂, x; agg=sum) + β * klfromgaussian(μ, logvar) + α * lastlayercoherence(lastlayer) + λ * l2reg(pars)
end

function VAEloss_boundedcoherence(x, F, model::FullVae, pars, lastlayer, β, λ, α)

    intermediate = model.encoder.encoderbody(x)
    μ = model.encoder.splitedμ(intermediate)
    logvar = model.encoder.splitedlogvar(intermediate)
    z = μ .+ randn(Float32, size(logvar)) .* exp.(0.5f0 .* logvar)
    x̂ = model.decoder(z)

    klgaussian = 0.5 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0))
    l2reg = sum(x -> sum(abs2, x), pars)
    boundedcoherence = log(sum(exp, (sum(abs2, F * lastlayer, dims=2)))) #taking the softmax

    logitbinarycrossentropy(x̂, x; agg=sum) + β * klgaussian + α * boundedcoherence + λ * l2reg
end

function VAEloss_boundedcoherence(x, F, model::FullVae, pars, lastlayer, β, λ, α, tblogger)

    intermediate = model.encoder.encoderbody(x)
    μ = model.encoder.splitedμ(intermediate)
    logvar = model.encoder.splitedlogvar(intermediate)
    z = μ .+ randn(Float32, size(logvar)) .* exp.(0.5f0 .* logvar)
    x̂ = model.decoder(z)

    klgaussian = 0.5 * sum(@. (exp(logvar) + μ^2 - logvar - 1.0))
    l2reg = sum(x -> sum(abs2, x), pars)
    boundedcoherence = log(sum(exp, (sum(abs2, F * lastlayer, dims=2)))) #taking the softmax

    with_logger(tblogger) do
        @info "loss terms" logitbinarycrossentropy(x̂, x; agg=sum) β * klgaussian α * boundedcoherence λ * l2reg
    end
end
