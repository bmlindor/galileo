# Julia v1.6
using CALCEPH,DelimitedFiles,Random,LinearAlgebra

include("regress.jl")
#include("CGS.jl")
# Load JPL ephemerides from data and set units
eph = Ephem("jup365.bsp") ; prefetch(eph)
options = useNaifId+unitKM+unitDay # useNaifId + unitDay + unitAU
AU = 149597870.700 #km
Random.seed!(42)
# Find when body_id transits between jd1 and jd2 for observer at n_obs with N orbit integration steps 
function find_transit(body_id::Int,eph::CALCEPH.Ephem,jd1::Float64,jd2::Float64,n_obs::Vector{Float64},N::Int)
  JD_0 = 0.0
  ff = zeros(N)
  xdotn = 0.0
  pos = zeros(3,N) 
  # Compute functions of position and velocity, f(t)=dot(x_bar_sky, v_bar_sky) and f'(t):
  function calc_ffs(t)
    pva = compute(eph,JD_0,t,body_id,naifId.id[:jupiter],options,2)./AU
    #println(JD_0)
    r = pva[1:3]; v = pva[4:6]; a = pva[7:9];
    f = dot(r,v) - (dot(r,n_obs))*(dot(v,n_obs))
    Df = dot(v,v) + dot(r,a) - (dot(v,n_obs))^2 - (dot(r,n_obs)*dot(a,n_obs))
    return f,Df,dot(r,n_obs),r
  end
  # Compute minimum sky separation of planet/moon wrt star/planet for all JDs
  dt =  (jd2 - jd1)/(N-1)
  JD = zeros(N)
  i_min = 1
  ff_min = Inf
  for i=1:N
    JD[i] = jd1 + dt*(i-1)
    JD_0 = JD[i]
    ff[i],Df,rdotn,pos[:,i] = calc_ffs(0.0)
  # Estimate of transit time when f(t)== 0:
      # Df > 0 for transit occuring; 
      # xdotn > 0 for planet in front of star as seen by observer; 
      # local minimum value over entire range (i.e. when close to zero)
    if (Df > 0) && (rdotn > 0) && (abs(ff[i]) < ff_min)
      i_min = i 
      ff_min = abs(ff[i])
    end
  end
   #println("Estimated Transit Time: ",JD[i_min])
  # Refine initial guess using linear approx: 
  JD_0 = JD[i_min]
  JD_n = 0.0
  JD_n1 = JD_n + 1
  JD_n2 = JD_n + 2
  iter = 0
  # ITMAX = 20
  ITMAX = 6 
  # we've found that we don't need large ITMAX to find solution; does that change for different bodies?
  for iter=0:ITMAX
      JD_n2 = JD_n1
      JD_n1 = JD_n
      while JD_n > 1
          JD_n -= 1.0
          JD_0 += 1.0
      end
      while JD_n < 0
          JD_n += 1.0
          JD_0 -= 1.0
      end
      f_n,Df_n,xdotn,x = calc_ffs(JD_n)
      JD_n -= f_n/Df_n 
      # Break out if we have reached maximum iterations,or if
      # current transit time estimate equals one of the prior two steps:
      if (JD_n == JD_n1) || (JD_n == JD_n2)
          break
      end
  end          
  JD_tt = JD_0 + JD_n
  #println("Refined Transit Time: ",JD_tt)
  # return JD_tt
	return JD,ff,i_min,pos,JD_tt
end
# Find the transit times for body_id, given planetary period estimate,and number of refinement steps N
function transit_times(body_id::Int,eph::CALCEPH.Ephem,t0,period::Float64,period_err::Float64,n_obs::Vector{Float64},N::Int)
  TT = Float64[]
  t_final = t0[end]
  # Initialize & find first transit time:
  JD,ff,i_min,pos,JD_tt = find_transit(body_id,eph,t0[1],t0[1]+period,n_obs,1000) 
  push!(TT,JD_tt)
  nt=1
  # Find subsequent transit times by shifting time frame by 1 planetary period:
  while JD_tt < t_final
    t_start = JD_tt+period-period_err
    t_end = JD_tt+period+period_err
    JD,ff,i_min,pos,JD_tt = find_transit(body_id,eph,t_start,t_end,n_obs,N)
    if (JD_tt>t_final) # last run of while loop doesn't meet condition , so need to break 
      break
    else
      push!(TT,JD_tt)
      nt+=1
    end
  end
  return TT
end
# Add Gaussian sigma noise to transit times
function fixed_noise(tt::Vector{Float64},sigma::Real)
  if sigma > 0
      sigtt = ones(length(tt)) * sigma / (24 * 3600) # sigma in seconds,sigtt in days
      noise = randn(length(tt)) .* sigtt  
      # println("Noise added with σ of ",string(sigma)," seconds.")
  else
      sigtt=0
      println("No noise added.")
  end
  return sigtt 
end
# Do linear regression of transit times, given mean orbital period
function linear_fit(tt::Vector{Float64},period::Float64,sigtt::Vector{Float64})
  nt=length(tt)
  noise = zeros(nt)
  x = zeros(2,nt)
  x[1,1:nt] .= 1.0
  x[2,1] = 0.0 
  for i=2:nt
    # currently accounts for missing transits (noncontinuous) 
    # by rounding [difference in consecutive transit times/Period]
      x[2,i] = round((tt[i]-tt[1])/period) 
  end
  # println(tt,sigtt,std(sigtt))
  # coeff[1] is best linear fit approx of first tt,coeff[2] is average period
  coeff,covcoeff = regress(x,tt,sigtt)
  t0,per=coeff[1],coeff[2]
  return x,t0,per
end
# Collect linear transit times (i.e. t_calc), given mean orbital period
function linear_times(tt::Vector{Float64},period::Float64,sigtt::Vector{Float64})
  nt=length(tt)
  x,t0,per=linear_fit(tt,period,sigtt)
  times=collect(t0 .+ per .* range(0,stop = nt-1,length = nt)) 
  return times
end
# Compute unit vector which intersects orbital plane of objects 1 and 2:
function calc_obs_loc(pos1,vel1,pos2,vel2)
  h1 = cross(pos1,vel1)
  h2 = cross(pos2,vel2)
  n_obs = cross(h2,h1)
  n_obs /= norm(n_obs) #from one direction when both transit
end
