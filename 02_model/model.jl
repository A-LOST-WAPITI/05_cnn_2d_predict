using CUDA
CUDA.device!(0)
using JLD2
using Flux
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using Random
using DelimitedFiles
using StatsBase
using Statistics
using GeometricFlux
using Flux.OneHotArrays: onehotbatch
using ProgressMeter
using MLUtils


Base.@kwdef mutable struct Args
    feature_num = 5     ##
    η = 3e-4            ## learning rate
    λ = 0             ## L2 regularizer param, implemented as weight decay
    batchsize = 32     ## batch size
    train_ratio = 0.9   ## ratio of data used in training procedure
    epochs = 200000        ## number of epochs
    seed = 0            ## set seed > 0 for reproducibility
    use_cuda = true     ## if true use cuda (if available)
    infotime = 1 	    ## report every `infotime` epochs
    checktime = 5       ## Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true     ## log training with tensorboard
    savepath = "runs/"  ## results path
end

struct WeightedPool{T}
    W::AbstractMatrix{T}
end
Flux.@functor WeightedPool
function WeightedPool(feature_num::Int; init=Flux.glorot_uniform)
    return WeightedPool(init(feature_num, feature_num))
end
(l::WeightedPool)(x::AbstractMatrix) = mean(l.W * x, dims=2)
(l::WeightedPool)(x::AbstractVector) = l.(x)


struct SuperCell
    fg::AbstractFeaturedGraph
    nf_vec::AbstractVector
end
Flux.@functor SuperCell

struct SuperCGConv{A, B}
    layers::AbstractVector{CGConv{A, B}}
    norm
    σ
end
Flux.@functor SuperCGConv

function SuperCGConv(dims::NTuple{2,Int}; init=Flux.glorot_uniform, bias=true)
    return SuperCGConv(
        [CGConv(dims, init=init, bias=bias) for _ = 1:4],
        BatchNorm(dims[1]),
        softplus
    )
end

function (m::SuperCGConv)(x::SuperCell)
    new_nf_vec = [(m.layers[index](x, index)) |> m.σ for index in eachindex(m.layers)]

    return SuperCell(x.fg, new_nf_vec)
end
(m::SuperCGConv)(x::AbstractVector{SuperCell}) = m.(x)

struct Split{T}
    paths::T
end
Flux.@functor Split

Split(paths...) = Split(paths)
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

(l::GraphParallel)(x::SuperCell) = SuperCell(l(x.fg), x.nf_vec)
(l::GraphParallel)(x::Vector{SuperCell}) = l.(x)

function rearrange(nf_vec, index)
    x_i = 2 - index%2
    y_i = (index + 1)÷2

    isone(x_i) && (x_p = 2; x_n = 2; true) || (x_p = 1; x_n = 1; true)
    isone(y_i) && (y_p = 2; y_n = 2; true) || (y_p = 1; y_n = 1; true)

    index_vec = [
        2(x_p - 1) + y_p,
        2(x_p - 1) + y_i,
        2(x_p - 1) + y_n,
        2(x_i - 1) + y_p,
        2(x_i - 1) + y_i,
        2(x_i - 1) + y_n,
        2(x_n - 1) + y_p,
        2(x_n - 1) + y_i,
        2(x_n - 1) + y_n
    ]

    return hcat(nf_vec[index_vec]...)
end
function get_center(nf_mat)
    atom_count = size(nf_mat, 2)
    atom_per_cell = atom_count ÷ 9

    start = 4atom_per_cell + 1
    stop = 5atom_per_cell

    return nf_mat[:, start:stop]
end

function (l::CGConv)(s_cell::SuperCell, index)
    nf = rearrange(s_cell.nf_vec, index)
    temp_fg = ConcreteFeaturedGraph(s_cell.fg, nf=nf)

    result_nf = get_center(node_feature(l(temp_fg)))

    return result_nf
end

super_mean(x::SuperCell) = mean(x.nf_vec)
super_mean(x::AbstractVector{SuperCell}) = super_mean.(x)

function z_score_norm!(x!::AbstractMatrix{T}) where T
    norm_vec = mean(x!, dims=2)   # 均值
    std_vec = std(x!, dims=2)     # 标准差

    for (x̄, sigma, feature) in zip(norm_vec, std_vec, eachrow(x!))
        feature .= @. (feature - x̄)/sigma
    end

    return norm_vec, std_vec
end

num_params(model) = sum(length, Flux.params(model))

round4(x) = round(x, digits=4)

function loss(model, x, y)
    ŷ = model(x)

    los = Flux.logitcrossentropy(ŷ[1], y[1]) +
        Flux.logitcrossentropy(ŷ[2], y[2]) +
        Flux.logitcrossentropy(ŷ[3], y[3]) +
        9 * Flux.mse(ŷ[4], y[4])
    return los/12
    # los = Flux.mse(ŷ[4][1, :], y[4][1, :])
    # return los
end
function eval_loss(loader, model, device)
    l = 0f0
    ntot = 0
    for (x_batch, y_batch...) in loader
        x_batch, y_batch = x_batch |> device, y_batch |> device
        item_count = length(x_batch)

        l += loss(model, x_batch, y_batch) * item_count
        ntot += item_count
    end
    return l/ntot |> round4
end

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()

    if use_cuda
        CUDA.allowscalar(false)
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    (
        id_vec,
        graph_vec,
        nf_vec,
        target_mat
    ) = load(
        "ready.jld2",
        "id",
        "graph_vec",
        "node_feature_vec",
        "target_vec"
    )
    x_data = [
        SuperCell(
            fg,
            fill(nf, 4)
        ) for (fg, nf) in zip(graph_vec, nf_vec)
    ]
    thermo_dyn_data = onehotbatch(target_mat[1, :], 1:3)
    phon_dyn_data = onehotbatch(target_mat[2, :], 0:1)
    stiff_dyn_data = onehotbatch(target_mat[3, :], 0:1)
    s_vec_data = target_mat[4:end, :]
    z_score_norm!(s_vec_data)
    material_count = length(id_vec)
    shuffle_indics = randperm(material_count)
    cutoff = Int(round(material_count * args.train_ratio))
    train_loader = Flux.DataLoader(
        (
            x_data[shuffle_indics[1:cutoff]],
            thermo_dyn_data[:, shuffle_indics[1:cutoff]],
            phon_dyn_data[:, shuffle_indics[1:cutoff]],
            stiff_dyn_data[:, shuffle_indics[1:cutoff]],
            s_vec_data[:, shuffle_indics[1:cutoff]]
        ),
        batchsize=args.batchsize
    )
    test_loader = Flux.DataLoader(
        (
            x_data[shuffle_indics[cutoff + 1:end]],
            thermo_dyn_data[:, shuffle_indics[cutoff + 1:end]],
            phon_dyn_data[:, shuffle_indics[cutoff + 1:end]],
            stiff_dyn_data[:, shuffle_indics[cutoff + 1:end]],
            s_vec_data[:, shuffle_indics[cutoff + 1:end]]
        ),
        batchsize=args.batchsize
    )
    train_batch_count = MLUtils.numobs(train_loader)
    @info "Dataset loaded"

    model = Chain(
        # GraphParallel(
        #     edge_layer=BatchNorm(12)
        # ),
        SuperCGConv((92, 12)),
        SuperCGConv((92, 12)),
        # GraphParallel(
        #     edge_layer=BatchNorm(12)
        # ),
        SuperCGConv((92, 12)),
        super_mean,
        (x -> mean.(x, dims=2)),
        Flux.batch,
        Flux.flatten,
        Dense(92, 128),
        # BatchNorm(128),
        softplus,
        # Dropout(0.5),
        Dense(128, 64),
        # BatchNorm(64),
        softplus,
        # Dropout(0.5),
        Split(
            Dense(64, 3),
            Dense(64, 2),
            Dense(64, 2),
            Dense(64, 9)
        )
    ) |> device
    @info "ToyModel model: $(num_params(model)) trainable params"

    ps = Flux.params(model)

    opt = Flux.Optimiser(
        NAdam(),
        # AdaDelta(),
        WeightDecay(args.λ)
    )
    # opt = Adam()
    # opt = AdaDelta()

    ## LOGGING UTILITIES
    if args.tblogger 
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) ## 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end

    function report(epoch)
        Flux.testmode!(model)
        train_loss = eval_loss(train_loader, model, device)
        test_loss = eval_loss(test_loader, model, device)        
        println("Epoch: $epoch   Train: $(train_loss)   Test: $(test_loss)")
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss=train_loss
                @info "test"  loss=test_loss
            end
        end
    end

    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        p = Progress(train_batch_count)
        for (x_batch, y_batch...) in train_loader
            x_batch, y_batch = x_batch |> device, y_batch |> device
            gs = Flux.gradient(ps) do
                loss(model, x_batch, y_batch)
            end

            Flux.Optimise.update!(opt, ps, gs)
            next!(p)
        end

        # if epoch % 10 == 0 && opt.os[1].eta > 1e-5
        #     opt.os[1].eta *= 0.95
        # end
        
        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "model.jld2") 
            let model = cpu(model) ## return model to cpu before serialization
                save(
                    modelpath,
                    Dict(
                        "model" => model,
                        "epoch" => epoch
                    )
                )
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    train()
end