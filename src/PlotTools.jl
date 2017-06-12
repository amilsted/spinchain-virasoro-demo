
module PlotTools

using PyPlot

function pointexclusion1D(ys, range=0.04, xspacing=0.04)
    incluster = BitArray(length(ys))
    clusters = BitArray[]
    for j in 1:length(ys)
        if !incluster[j]
            incluster[j] = true
            c = BitArray(length(ys))
            c[j] = true
            for k in 1:length(ys)
                if !incluster[k] && any(abs(ys[c] - ys[k]) .< range)
                    incluster[k] = true
                    c[k] = true
                end
            end
            
            push!(clusters, c)
        end
    end
    xs = zeros(ys)
    for c in clusters
        clen = countnz(c)
        if clen > 1
            offs = Float64[j * xspacing for j in 0:clen-1]
            offs = offs - mean(offs)
            xs[c] = offs
        end
    end
    xs
end

function pointexclusion1D{T}(xs::Vector{T}, ys::Vector{T}, yrange=0.04, xspacing=0.04, x_round_digits=6)    
    xs_u = unique(round(xs, x_round_digits))
    
    xs_new = zeros(xs)
    
    for x in xs_u
        flt = abs(xs - x) .< 10.0^(-x_round_digits)
        ys_x = ys[flt]
        
        x_shifts = pointexclusion1D(ys_x, yrange, xspacing)
        xs_new[flt] = xs[flt] + x_shifts
    end
    
    xs_new, ys
end

function savepdf_cropped(fn, args...; kwargs...)
    savefig(fn, args...; kwargs...)
    run(`pdfcrop $fn $fn`)
end

function savepdf_cropped_compfonts(fn, args...; kwargs...)
    savefig(fn, args...; kwargs...)
    fntmp = fn * ".tmp"
    run(`pdfcrop $fn $fntmp`)
    run(`gs -sDEVICE=pdfwrite -dNOPAUSE -dQUIET -dBATCH -dCompressFonts=true -sOutputFile=$fn $fntmp`)
    run(`rm $fntmp`)
end

function savepdf_cropped_viaeps(fn, args...; kwargs...)
    fneps = fn * ".eps"
    savefig(fneps, args...; kwargs...)
    fnpdf = fn * ".pdf"
    run(`epstopdf $fneps $fnpdf`)
    run(`pdfcrop $fnpdf $fnpdf`)
end

end