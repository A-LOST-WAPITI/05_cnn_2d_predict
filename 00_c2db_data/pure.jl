using DelimitedFiles
using JSON
using HTTP
using JLD2
using ProgressMeter
using Dates
using Distributed


const PRE_URL = "https://cmrdb.fysik.dtu.dk/c2db/row/"
const HULL_JSON_URL = "/data/results-asr.convex_hull.json/json"
const STIFFNESS_JSON_URL = "/data/results-asr.stiffness.json/json"
const STRUCTURE_JSON_URL = "/data/structure.json/json"

function julia_main()
    id_dict = load("material_id.jld2")
    json_url_vec = [HULL_JSON_URL, STIFFNESS_JSON_URL, STRUCTURE_JSON_URL]
    json_type_vec = ["hull", "stiffness", "structure"]

    result_dict = Dict{String, Any}()
    for config in ["ph_sh", "ph_sl", "pl_sh", "pl_sl"]
        @info "Produce $(config) materials."

        id_vec = id_dict[config]
        for material_id in id_vec
            result_dict[material_id] = Dict{String, Any}("config" => config)
        end

        for (json_type, json_url) in zip(json_type_vec, json_url_vec)
            @info "Produce $(json_type)."

            @showprogress @distributed for material_id in id_vec
                r = HTTP.request(
                    "GET",
                    PRE_URL * material_id * json_url,
                    connection_limit=32,
                    retry=true,
                    retries=5
                )

                result_dict[material_id][json_type] = JSON.parse(r.body |> String)
            end
        end
    end

    jldsave(
        "C2DB.jld2";
        data = result_dict,
        time = today()
    )
end


if abspath(PROGRAM_FILE) == @__FILE__
    julia_main()
end