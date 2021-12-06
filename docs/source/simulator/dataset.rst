Dataset
=======

Our VMAgent collects the **real scheduling data in huawei cloud** for one month. The
`dataset <https://github.com/mail-ecnu/VMAgent/blob/master/vmagent/data/dataset.csv>`__ is placed in our repository.

Statistical Analysis
--------------------

The statsical information of the dataset is listed below.

================== =========================== =========================== ============= ===============
Number of VM types Number of creation requests Number of deletion requests Time duration Server location
================== =========================== =========================== ============= ===============
15                 125430                      116313                      30 Days       East China
================== =========================== =========================== ============= ===============

To gain better understanding of the cpu and memory distribution, we plot the histograms of the cpu and memory.

.. figure:: ../images/scenarios/cpu.png
   :alt: cpu

.. figure:: ../images/scenarios/mem.png
   :alt: mem

More than 2/3 requests only consumes 1U and less than 2G. We also plot the statiscs of the (cpu, mem) request:

.. figure:: ../images/scenarios/vm_type.png
   :alt: vmtype

The 1U1G,1U2G, 2U4G and 4U8G constitues the main body of the requests.

We also visualize the dynamic of virtual machine during the month:

.. figure:: ../images/scenarios/alive_vms.png
   :alt: alives

Although there exists deletion request, the number of alive virtual machines increses from 0 to more than 8000. It
should be noted that, even in the one month, the VM’s dynamic is highly related to the time.
``Increase, Flux, Increase, Flux`` happens through the one month.

.. figure:: ../images/scenarios/cpu_mem.png
   :alt: cpu-mem

We also visualize the allocated cpu and memory dynamic above. They can be helpful in constructing domain knowledge.

Naive Baselines performance
---------------------------

Another way to describe the dataset is measuring performance of naive baselines in the dataset. We adopt First-Fit and
Best-Fit as the naive baselines and conduct experiments on different settings.

We conduct fading and recovering experiments with 5, 20, 50 servers and each server has 40 cpu and 90 memeory.

========== ================= ======== ===================== ========================= =========================
Scenario   Number of servers Method   Number of Allocations Terminated CPU Rate       Terminated MEM Rate
========== ================= ======== ===================== ========================= =========================
Fading     5                 BestFit  :math:`211.7 \pm 30`  :math:`91.6\% \pm 9.4\%`  :math:`83.6\% \pm 9.2\%`
\                            FirstFit :math:`224.5 \pm 28`  :math:`98.3\% \pm 1.9\%`  :math:`90.0\% \pm 1.9\%`
\          20                BestFit  :math:`735.1 \pm 83`  :math:`63.5\% \pm 29.2\%` :math:`35.7\% \pm 21.9\%`
\                            FirstFit :math:`888.0 \pm 65`  :math:`91.6\% \pm 8.5\%`  :math:`64.7 \pm 5.6\%`
\          50                BestFit  :math:`1674.5 \pm 28` :math:`91.6\% \pm 1.1\%`  :math:`84.3 \pm 1.0\%`
\                            FirstFit :math:`2298.3 \pm 19` :math:`95.5\% \pm 0.7\%`  :math:`91.5\% \pm 0.5\%`
Recovering 5                 BestFit  :math:`221.1 \pm 29`  :math:`96.3\% \pm 5.6\%`  :math:`88.1\% \pm 5.7\%`
\                            FirstFit :math:`222.7 \pm 27`  :math:`97.2\% \pm 3.4\%`  :math:`89.0\% \pm 3.4\%`
\          20                BestFit  :math:`850.0 \pm 13`  :math:`99.1\% \pm 0`      :math:`95.8\% \pm 0`
\                            FirstFit :math:`926.1 \pm 10`  :math:`98.7\% \pm 0.5\%`  :math:`96.5\% \pm 0.3\%`
\          50                BestFit  :math:`1829.6 \pm 37` :math:`92.8\% \pm 1.4\%`  :math:`88.8\% \pm 0.2\%`
\                            FirstFit :math:`2301.7 \pm 19` :math:`95.0\% \pm 0.5\%`  :math:`91.1\% \pm 0.4\%`
========== ================= ======== ===================== ========================= =========================
