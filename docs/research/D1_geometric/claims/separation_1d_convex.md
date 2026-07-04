# separation_1d_convex

**status:** proved ($d=1$)

**statement.** $\sigma=\mathrm{ReLU}$, $d=1$, $\Omega=[-1,1]$, $V=\tfrac{C}{2}x^2-g$ with $g$ **general
convex**, gradient-$L^2$ best-$n$-term cost. Then both models converge at rate $n^{-1}$,
$$
e_{\text{semi}}(n)\sim \frac{K_{\text{semi}}}{n},\quad e_{\text{signed}}(n)\sim \frac{K_{\text{signed}}}{n},
\qquad
\boxed{\ \frac{K_{\text{semi}}}{K_{\text{signed}}}=\frac{\int|g''|}{\int|C-g''|}=\frac{\int_\Omega|V_{qq}''|\ \text{of }g}{\int_\Omega|V''|}\ }
$$
where $\int|g''|=\mathrm{bud}(g)$ and $\int|C-g''|=\int|V''|$. Hence
$$
N(\text{semiconcave},V,\varepsilon)\ <\ N(\text{signed},V,\varepsilon)\quad\Longleftrightarrow\quad \int|g''|<\int|V''|,
$$
i.e. **the semiconcave model is more sparse iff the convex correction $g$ carries less total
curvature than $V$ itself.** If $g'$ is finitely-stepped ($g$ piecewise-linear), $e_{\text{semi}}=0$
at finite $n$ → *divergent* separation (`separation_flat` is this sub-case).

**proof.** In 1-D a signed ReLU sum's derivative is an arbitrary $n$-step piecewise-constant
function; the semiconcave model's derivative is $C x + (n\text{-step})$ [the head reproduces
$Cx$ exactly]. So $e_{\text{signed}}(n)=$ best $n$-step $L^2$ approximation error of $V'=Cx-g'$,
and $e_{\text{semi}}(n)=$ best $n$-step error of $g'$ [head removes $Cx$]. The free-knot best
$n$-step $L^2$ error of a function $h$ is $\sim \tfrac{1}{2\sqrt3\,n}\int|h'|$. With $h=g'$
($\int|g''|=\mathrm{bud}(g)$) and $h=V'$ ($\int|V''|=\int|C-g''|$) the displayed ratio follows; the
$n^{-1}$ rate and ratio are confirmed exactly by DP-optimal $n$-step approximation
(`../../scripts/sep_1d_convex_check.py`: uniform $g''{=}C/2$ → ratio $1.000$; $g''{=}0.9C$ →
$9.000$; flat $g$ → $0$). $\square$

**sufficient conditions (clean).** If $V$ is convex ($g''\le C$): $\int|V''|=2C-\mathrm{bud}(g)$, so
more-sparse $\iff \mathrm{bud}(g)<C$ ("$g$'s total curvature below the semiconcavity constant",
i.e. the quadratic head dominates). Sharpest reading: **more sparse $\iff$ $V$ is more curved
overall than its convex correction** — true when $V$ has a dominant quadratic bulk, false when
$V$ is nearly linear (weak bulk) so the fixed head $\tfrac{C}{2}x^2$ is wasteful.

**uses.** `head_reduction` (head removes $Cx$); `signed_lower_bound` ($n^{-1}$ floor).

**caveat / honesty.** This *refines* "more sparse" — it is **not unconditional**: for weakly-
curved $V$ (large $\mathrm{bud}(g)$ relative to $C$) the signed model is more sparse, because it
exploits cancellation $Cx-g'$ that the fixed nonneg head cannot. The program's premise (HJB
value functions have a dominant curved bulk) is exactly the regime where semiconcave wins.

**attempts.** 1-D is clean (signed deriv = any $n$-step). $d\ge2$ → `separation_general`:
same *structure* (compare $\int|D^2g|$ vs $\int|D^2V|$) but needs (i) cone-free in $d\ge2$
(`cone_free_convex`: $\gamma^+(g)=\gamma(g)$ for convex $g$), and (ii) the $d\ge2$ convex-function gradient
approximation rate (may be $n^{-2/d}$, not $n^{-1}$ — Gruber polytopal). Open.
