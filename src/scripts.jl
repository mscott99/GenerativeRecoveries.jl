function addreshape_tomodel(model)
    new_decoder = Chain(model.decoder..., x -> reshape(x, 28, 28))
    new_model = FullVae(model.encoder, new_decoder)
end