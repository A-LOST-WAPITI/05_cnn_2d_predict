using JLD2
using LinearAlgebra
using JSON
using TOML
using Statistics
using InvertedIndices
using GeometricFlux
using SparseArrays


const C2DB_DATA_PATH = "../00_c2db_data/data.jld2"
const ELEMENT_DATA_PATH = "../00_c2db_data/FixedElementsData.json"


"""
Transform a regular cartesian index `A[i, j]` into a CSC-compatible index `spA.nzval[idx]`.
"""
SparseArrays.getcolptr(S::AbstractSparseArray, col::Integer) = S.colptr[col]:(S.colptr[col+1]-1)
SparseArrays.getcolptr(S::AbstractSparseArray, I::UnitRange) = S.colptr[I.start]:(S.colptr[I.stop+1]-1)
function get_csc_index(S::AbstractSparseArray, i::Integer, j::Integer)
    idx1 = SparseArrays.getcolptr(S, j)
    row = view(rowvals(S), idx1)
    idx2 = findfirst(x -> x == i, row)
    return idx1[idx2]
end

"""
    _ZScoreNormalization!(X!::AbstractArray{T}) where T

用于进行Zero-Score标准化的函数。
允许`X!`中存在`nothing`，在计算过程中会被忽略。
"""
function _ZScoreNormalization!(X!::AbstractArray{T}) where T
    NothingIndexArray = findall(isnothing, X!)  # 找到所有的nothing的位置
    NormalX! = view(X!, Not(NothingIndexArray)) # 引用所有非nothing成数组

    X̄ = NormalX! |> mean    # 均值
    σ = NormalX! |> std     # 标准差

    NormalX! .= @. (NormalX! - X̄)/σ # 自身更改
end

function julia_main()
    (
        id_vec,
        dis_mat,
        dyn_mat,
        number_vec,
        stiff_mat,
        cell_mat_vec,
        pos_mat_vec
    ) = load(
        C2DB_DATA_PATH,
        "id_vec",
        "dis_mat",
        "dyn_mat",
        "number_vec",
        "stiff_mat",
        "cell_mat_vec",
        "pos_mat_vec"
    )
    element_dict = JSON.parsefile("atom_init.json")

    material_count = length(id_vec)
    material_graph_vec = Vector{FeaturedGraph}(undef, material_count)
    nf_mat_vec = Vector{Matrix{Float32}}(undef, material_count)
    target_mat = zeros(Float32, 12, material_count)
    for material_index = 1:material_count
        cell_mat = cell_mat_vec[material_index]
        pos_mat = pos_mat_vec[material_index]
        numbers = number_vec[material_index]
        atom_count = length(numbers)

        adj_mat = [
            i == j || ((i - 1) ÷ atom_count == 4) || ((j - 1) ÷ atom_count == 4) 
            for i = 1:9*atom_count, j = 1:9*atom_count
        ]
        dis_mat = fill(Inf32, 9atom_count, 9atom_count)
        for i_index = 1:9atom_count, j_index = 1:9atom_count
            i_x_pad = (i_index÷atom_count)÷3 - 1
            i_y_pad = (i_index÷atom_count)%3 - 1
            j_x_pad = (i_index÷atom_count)÷3 - 1
            j_y_pad = (i_index÷atom_count)%3 - 1
            i_i_index = (i_index - 1)%atom_count + 1
            i_j_index = (j_index - 1)%atom_count + 1

            if adj_mat[i_index, j_index]
                dis_mat[i_index, j_index] = norm(
                    (
                        pos_mat[:, i_i_index] .+
                        i_x_pad .* cell_mat[:, 1] .+
                        i_y_pad .* cell_mat[:, 2]
                    ) .- (
                        pos_mat[:, i_j_index] .+
                        j_x_pad .* cell_mat[:, 1] .+
                        j_y_pad .* cell_mat[:, 2]
                    )
                )
            end
        end
        min_12_indics = [sortperm(dis_mat[4atom_count + index, :])[1:13] for index = 1:atom_count]
        min_12_mat = falses(9atom_count, 9atom_count)
        for i_index = 1:9atom_count, j_index = 1:9atom_count
            if (i_index - 1)÷atom_count == 4 && j_index in min_12_indics[i_index - 4atom_count]
                min_12_mat[i_index, j_index] = true
            end
            if (j_index - 1)÷atom_count == 4 && i_index in min_12_indics[j_index - 4atom_count]
                min_12_mat[i_index, j_index] = true
            end
        end
        adj_mat = [i == j for i = 1:9atom_count, j = 1:9atom_count] .|| (adj_mat .&& min_12_mat)
        temp_fg = FeaturedGraph(adj_mat)
        ef_mat = zeros(Float32, 12, temp_fg.graph.E)
        ef_vec = zeros(Float32, temp_fg.graph.E)
        for i_index = 1:9atom_count, j_index = 1:9atom_count
            if adj_mat[i_index, j_index]
                ef_index = temp_fg.graph.edges[get_csc_index(temp_fg.graph.S, i_index, j_index)]
                ef_vec[ef_index] = dis_mat[i_index, j_index]
            end
        end
        for (dis_index, dis) in enumerate(ef_vec)
            for pow_index = 1:12
                pow = pow_index - 13
                ef_mat[pow_index, dis_index] = iszero(dis) ? one(dis) : dis^(pow)
            end
        end

        fg = ConcreteFeaturedGraph(temp_fg, ef=ef_mat)
        material_graph_vec[material_index] = fg

        nf_mat_vec[material_index] = reduce(
            hcat,
            [
                Vector{Float32}(element_dict[number |> repr])
                for number in numbers
            ]
        )
        target_mat[1:3, material_index] .= dyn_mat[:, material_index]
        target_mat[4:end, material_index] .= vec(stiff_mat[:, :, material_index])
    end
    save(
        "ready.jld2",
        Dict(
            "id" => id_vec,
            "graph_vec" => material_graph_vec,
            "node_feature_vec" => nf_mat_vec,
            "target_vec" => target_mat
        )
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    julia_main()
end
