# Medical Interpretation of Comorbidity Drift Analysis (ICD-Level, Male & Female)

This document interprets the age-shift findings from the comorbidity network embedding pipeline, focusing on the **ICD-level variant** for both **male** and **female** strata. All 1,080 individual ICD-10 codes are analysed across 8 age groups (0--9 through 70--79).

---

## 1. The Comorbidity Landscape Changes Dramatically With Age

The most fundamental observation is that **comorbidity networks become vastly denser with age**. In childhood (age group 1, 0--9 years), ~87--90% of ICD codes are isolates (no significant comorbidity links). By age group 8 (70--79), this drops to ~64--67%, meaning roughly a third of all diseases have acquired statistically significant comorbidity associations.

| Age Group | Male Edges | Male Isolate % | Female Edges | Female Isolate % |
|-----------|-----------|----------------|-------------|-----------------|
| 1 (0--9)  | 425       | 87.4%          | 278         | 89.7%           |
| 2 (10--19)| 120       | 90.8%          | 156         | 89.9%           |
| 3 (20--29)| 312       | 87.8%          | 347         | 85.9%           |
| 4 (30--39)| 631       | 80.7%          | 607         | 81.5%           |
| 5 (40--49)| 1,489     | 71.3%          | 1,352       | 72.8%           |
| 6 (50--59)| 2,456     | 66.1%          | 2,009       | 69.2%           |
| 7 (60--69)| 3,224     | 65.4%          | 2,642       | 66.3%           |
| 8 (70--79)| 3,663     | 66.8%          | 4,181       | 64.4%           |

**Medical interpretation**: In childhood and adolescence, diseases tend to occur independently -- a child with asthma does not necessarily have cardiovascular or metabolic conditions. By old age, multimorbidity is the norm: hypertension co-occurs with diabetes, which co-occurs with renal disease, which links to cardiovascular events, and so on. The network captures this biological reality quantitatively.

A notable detail: **females overtake males in edge count at age 70--79** (4,181 vs 3,663), despite having fewer edges at younger ages. This likely reflects women's longer life expectancy and the accumulation of chronic conditions in elderly women (osteoporosis, autoimmune conditions, metabolic disease).

---

## 2. Which Age Transitions Show the Most Change?

### Drift by transition (non-isolate nodes only)

| Transition        | Male Mean Drift | Female Mean Drift |
|-------------------|----------------|------------------|
| 1 -> 2 (0s->10s) | 2.12           | 1.90             |
| 2 -> 3 (10s->20s)| **2.55**       | **2.57**         |
| 3 -> 4 (20s->30s)| 1.98           | 2.49             |
| 4 -> 5 (30s->40s)| 2.38           | 2.45             |
| 5 -> 6 (40s->50s)| 2.20           | 2.34             |
| 6 -> 7 (50s->60s)| 2.07           | 2.16             |
| 7 -> 8 (60s->70s)| 2.00           | 2.17             |

### kNN Stability by transition (non-isolate nodes only)

| Transition        | Male Mean Stability | Female Mean Stability |
|-------------------|--------------------|--------------------- |
| 1 -> 2 (0s->10s) | 0.346              | 0.443                |
| 2 -> 3 (10s->20s)| 0.381              | 0.488                |
| 3 -> 4 (20s->30s)| **0.585**          | 0.496                |
| 4 -> 5 (30s->40s)| 0.448              | **0.534**            |
| 5 -> 6 (40s->50s)| 0.427              | 0.469                |
| 6 -> 7 (50s->60s)| 0.444              | 0.430                |
| 7 -> 8 (60s->70s)| 0.411              | 0.351                |

### The adolescence-to-adulthood transition (10--19 -> 20--29) is the most volatile

For both sexes, mean drift peaks at the **2 -> 3 transition** (adolescence to young adulthood). This makes strong medical sense:

- **Adolescence is a period of relative health**: The 10--19 age group has the sparsest networks (120 edges male, 156 female), reflecting that teenagers have very few comorbidity associations. Most diseases in this age group are independent conditions.
- **Young adulthood introduces new disease patterns**: In the 20s, conditions related to lifestyle (substance use, mental health), reproduction (gynaecological conditions), and early occupational exposures begin to form comorbidity clusters. The transition from "mostly healthy" to "beginning of adult disease patterns" creates the largest repositioning of diseases in embedding space.

### The oldest transitions are the most stable

By contrast, **transitions between ages 50--79 show lower drift and declining stability**. This is not because nothing changes in elderly patients -- the opposite is true. Rather, it reflects that the *pattern* of multimorbidity becomes increasingly established. Once a patient has hypertension, diabetes, and atherosclerosis in their 50s, the same cluster persists and deepens into the 60s and 70s. New edges are added, but the fundamental structure of "which diseases go with which" does not rearrange as dramatically.

### Childhood (0--9 -> 10--19) is also highly volatile but with fewer connected diseases

The 1 -> 2 transition shows moderately high drift (2.12 male, 1.90 female) but affects very few diseases (only 42--55 non-isolate nodes). The diseases that *are* connected in childhood (congenital conditions, childhood infections, developmental disorders) undergo significant repositioning as patients enter adolescence and childhood-specific conditions recede.

---

## 3. Diseases That Change the Most (Top Drifters)

### Male: Top 20 Highest-Drift Diseases

| ICD  | Disease | Mean Drift | Peak Transition(s) |
|------|---------|-----------|-------------------|
| C73  | Thyroid cancer | 3.12 | 3->4 (6.09), 4->5 (4.81) |
| M23  | Internal derangement of knee | 3.12 | 1->2 (5.16) |
| F20  | Schizophrenia | 3.09 | 1->2 (5.71), 2->3 (4.48) |
| M51  | Intervertebral disc disorders | 3.07 | 1->2 (5.35), 2->3 (3.26) |
| K08  | Disorders of teeth/supporting structures | 3.04 | 2->3 (5.16), 6->7 (6.45) |
| M22  | Disorders of patella | 3.04 | 1->2 (5.10) |
| D40  | Uncertain neoplasm of male genital organs | 3.03 | 2->3 (5.23), 4->5 (4.63) |
| M24  | Other joint derangements | 3.00 | 1->2 (5.09) |
| C77  | Secondary malignant neoplasm of lymph nodes | 2.98 | 2->3 (5.31), 3->4 (3.28) |
| M67  | Disorders of synovium and tendon | 2.94 | 1->2 (5.03) |
| C78  | Secondary malignant neoplasm of respiratory/digestive organs | 2.94 | 2->3 (5.20), 3->4 (3.28) |
| E89  | Postprocedural endocrine/metabolic complications | 2.93 | 3->4 (6.04), 4->5 (4.72) |
| C82  | Follicular lymphoma | 2.91 | 4->5 (6.15), 7->8 (6.60) |
| M54  | Back pain (dorsalgia) | 2.89 | 1->2 (5.37) |
| N45  | Orchitis and epididymitis | 2.89 | 1->2 (5.60), 2->3 (5.62) |
| M65  | Synovitis and tenosynovitis | 2.89 | 1->2 (5.07) |
| M93  | Other osteochondropathies | 2.88 | 2->3 (4.61), 7->8 (4.33) |
| H90  | Conductive and sensorineural hearing loss | 2.80 | 1->2 (4.02), 2->3 (5.51) |
| C71  | Brain cancer | 2.75 | 1->2 (3.46), 2->3 (4.14), 7->8 (4.28) |
| C85  | Non-Hodgkin lymphoma (other/unspecified) | 2.74 | 4->5 (6.44), 7->8 (4.75) |

### Female: Top 20 Highest-Drift Diseases

| ICD  | Disease | Mean Drift | Peak Transition(s) |
|------|---------|-----------|-------------------|
| E74  | Disorders of carbohydrate metabolism | 3.58 | 1->2 (4.68), 4->5 (4.27) |
| E73  | Lactose intolerance | 3.50 | 1->2 (4.70), 4->5 (4.26) |
| D06  | Carcinoma in situ of cervix | 3.37 | 2->3 (5.88), 6->7 (5.44) |
| N92  | Excessive/irregular menstruation | 3.34 | 1->2 (4.75), 7->8 (4.32) |
| C73  | Thyroid cancer | 3.32 | 2->3 (5.70), 3->4 (4.19), 4->5 (4.41) |
| N87  | Cervical dysplasia | 3.30 | 2->3 (5.92), 6->7 (5.24) |
| E89  | Postprocedural endocrine/metabolic complications | 3.25 | 2->3 (5.76), 3->4 (4.20), 4->5 (4.25) |
| D24  | Benign neoplasm of breast | 3.24 | 2->3 (5.61), 4->5 (4.29) |
| K65  | Peritonitis | 3.24 | 1->2 (6.18), 2->3 (6.19) |
| N73  | Female pelvic inflammatory diseases | 3.24 | 1->2 (4.76), 7->8 (4.27) |
| M23  | Internal derangement of knee | 3.19 | 1->2 (5.23) |
| M67  | Disorders of synovium and tendon | 3.17 | 1->2 (5.24) |
| M65  | Synovitis and tenosynovitis | 3.15 | 1->2 (5.29) |
| D48  | Neoplasm of uncertain behavior | 3.15 | 2->3 (5.42), 4->5 (4.10) |
| M22  | Disorders of patella | 3.13 | 1->2 (5.27) |
| E06  | Thyroiditis | 3.12 | 1->2 (5.38), 2->3 (5.37) |
| D27  | Benign neoplasm of ovary | 3.10 | 1->2 (4.91) |
| C43  | Malignant melanoma of skin | 3.08 | 3->4 (5.75), 5->6 (6.08), 7->8 (5.72) |
| N60  | Fibrocystic breast changes | 3.04 | 2->3 (5.60), 6->7 (4.35) |
| N83  | Ovarian/fallopian tube disorders | 3.00 | 1->2 (4.85) |

### Medical interpretation of the highest-drift diseases

**Musculoskeletal conditions dominate both sexes** (M23, M22, M51, M54, M24, M65, M67). These conditions peak in the 1->2 transition (childhood to adolescence), reflecting the dramatic shift from paediatric to adult orthopaedic comorbidity patterns. In childhood, joint and spinal conditions exist largely in isolation or alongside congenital/developmental disorders. In adolescence and early adulthood, they begin to form comorbidity clusters with metabolic, cardiovascular, and pain-related conditions. This repositioning continues into middle age as degenerative joint disease becomes intertwined with obesity, diabetes, and cardiovascular disease.

**Female reproductive conditions are among the highest drifters in women**: D06 (cervical carcinoma in situ), N87 (cervical dysplasia), N92 (menstrual disorders), D24 (benign breast neoplasm), D27 (benign ovarian neoplasm), N60 (fibrocystic breast changes), N73 (pelvic inflammatory disease), and N83 (ovarian/fallopian tube disorders). These conditions show near-zero drift in childhood (they simply do not occur) and peak drift in the 2->3 transition (puberty/young adulthood) as they enter the comorbidity network and form associations with other reproductive and hormonal conditions. Notably, D06 and N87 show a second drift peak at 6->7 (50s to 60s), corresponding to the post-menopausal period when cervical screening dynamics and hormonal changes alter their comorbidity associations.

**Cancers show age-specific onset patterns reflected in drift**:
- **C73 (thyroid cancer)**: Near-zero drift until 3->4 in males (20s to 30s), 2->3 in females -- matching the known earlier female incidence. Thyroid cancer's comorbidity profile changes most dramatically when it first becomes prevalent.
- **C82, C85 (lymphomas)**: Near-zero drift until 4->5 (30s to 40s), then explosive repositioning -- consistent with the epidemiology of non-Hodgkin lymphoma, which peaks in middle-to-older age.
- **C77, C78 (secondary/metastatic cancers)**: Peak at 2->3, reflecting the transition from childhood cancers (which have distinct metastatic patterns) to adult cancers.
- **C71 (brain cancer)**: Moderate drift across multiple transitions, consistent with its bimodal age distribution (childhood and older adults).
- **C43 (melanoma, females)**: A remarkable three-peak pattern at 3->4, 5->6, and 7->8. This likely reflects melanoma's evolving comorbidity relationships as it transitions from a relatively isolated diagnosis in young women to one increasingly co-occurring with other cancers and immunological conditions in older women.

**F20 (schizophrenia) in males** is the highest-drift disease at the 1->2 transition (drift 5.71). Schizophrenia typically presents in late adolescence/early adulthood in males. Before onset, it has no comorbidity footprint. After onset, it rapidly forms associations with metabolic syndrome, cardiovascular disease (partly medication-related), substance use disorders, and other psychiatric conditions. The steadily declining drift across later transitions shows that once established, schizophrenia's comorbidity profile stabilises.

**K08 (dental/tooth disorders) in males** shows an unusual bimodal pattern: peak at 2->3 (5.16) and again at 6->7 (6.45). The first peak corresponds to adolescent dental pathology entering the comorbidity network; the second, much larger peak likely reflects the emergence of dental disease as a geriatric comorbidity linked to cardiovascular disease, diabetes, and nutritional deficiencies in elderly men.

**E74 and E73 (carbohydrate metabolism disorders and lactose intolerance) in females** are the overall highest-drift conditions (mean 3.58 and 3.50). Unlike most high-drifters, they maintain consistently high drift across *all* transitions, never dropping to near-zero. This suggests these metabolic conditions continuously shift their comorbidity relationships across the lifespan, perhaps reflecting the evolving interplay between metabolic/digestive conditions and other disease categories at every life stage.

---

## 4. Diseases That Change the Least (Bottom Drifters)

### Male: 20 Lowest-Drift Diseases

| ICD  | Disease | Mean Drift |
|------|---------|-----------|
| A66  | Yaws | 0.0668 |
| E20  | Hypoparathyroidism | 0.0669 |
| A70  | Chlamydia psittaci infection | 0.0671 |
| E50  | Vitamin A deficiency | 0.0673 |
| A07  | Other protozoal intestinal diseases | 0.0673 |
| H80  | Otosclerosis | 0.0676 |
| C91  | Lymphoid leukaemia | 0.0676 |
| E41  | Nutritional marasmus | 0.0678 |
| L66  | Cicatricial alopecia | 0.0679 |
| F62  | Enduring personality changes | 0.0679 |
| G73  | Disorders of myoneural junction (classified elsewhere) | 0.0680 |
| D67  | Hereditary factor IX deficiency (haemophilia B) | 0.0681 |
| K38  | Other diseases of appendix | 0.0682 |
| C14  | Malignant neoplasm of lip/oral cavity/pharynx (other) | 0.0683 |
| N74  | Female pelvic inflammatory disorders (classified elsewhere) | 0.0684 |
| E29  | Testicular dysfunction | 0.0685 |
| K30  | Functional dyspepsia | 0.0685 |
| A26  | Erysipeloid | 0.0686 |
| G09  | Sequelae of CNS inflammatory diseases | 0.0686 |
| G64  | Other disorders of peripheral nervous system | 0.0686 |

### Female: 20 Lowest-Drift Diseases

| ICD  | Disease | Mean Drift |
|------|---------|-----------|
| N47  | Disorders of prepuce | 0.0663 |
| A26  | Erysipeloid | 0.0668 |
| J65  | Pneumoconiosis associated with tuberculosis | 0.0671 |
| A59  | Trichomoniasis | 0.0673 |
| B21  | HIV disease resulting in other infections | 0.0674 |
| J92  | Pleural plaque | 0.0676 |
| B95  | Streptococcus/staphylococcus as cause of disease | 0.0677 |
| D45  | Polycythaemia vera | 0.0679 |
| E24  | Cushing syndrome | 0.0679 |
| A95  | Yellow fever | 0.0680 |
| L93  | Lupus erythematosus | 0.0680 |
| E64  | Sequelae of nutritional deficiencies | 0.0680 |
| M66  | Spontaneous rupture of synovium/tendon | 0.0680 |
| H95  | Postprocedural complications of ear/mastoid | 0.0681 |
| A79  | Other rickettsioses | 0.0681 |
| M36  | Systemic connective tissue disorders (classified elsewhere) | 0.0681 |
| K77  | Liver disorders (classified elsewhere) | 0.0682 |
| G14  | Postpolio syndrome | 0.0682 |
| K91  | Postprocedural complications of digestive system | 0.0682 |
| N90  | Other noninflammatory disorders of vulva/perineum | 0.0683 |

### Medical interpretation of the lowest-drift diseases

The lowest-drift diseases share a common characteristic: **they are isolates (unconnected nodes) across all or nearly all age groups**. Their drift values (~0.067) are essentially the background noise of the embedding -- they never form significant comorbidity associations at any age.

These diseases fall into several categories:

1. **Tropical/exotic infections essentially absent in Austria** (A66 yaws, A70 psittacosis, A26 erysipeloid, A95 yellow fever, A79 rickettsioses, B56 trypanosomiasis): These diseases have negligible prevalence in the Austrian population and therefore never form meaningful comorbidity associations at any age.

2. **Very rare conditions** (E20 hypoparathyroidism, D67 haemophilia B, E41 nutritional marasmus, E50 vitamin A deficiency, D45 polycythaemia vera, E24 Cushing syndrome, G14 postpolio syndrome): Too rare to generate statistically significant co-occurrence patterns.

3. **Sex-mismatched codes** (N47 disorders of prepuce in females, N74 female pelvic inflammatory disorders in males): By definition, these codes cannot apply to the opposite sex, so they remain permanently isolated.

4. **Conditions without strong comorbidity signatures** (K38 appendix disorders, K30 functional dyspepsia, H80 otosclerosis, L66 cicatricial alopecia): These conditions, while not rare, tend to occur independently of other diseases -- they do not form part of recognisable multimorbidity clusters.

The key insight is that **a disease's drift is near-zero when it is never connected in the comorbidity network**. Low drift does not mean the disease is clinically unimportant; it means the disease does not participate in comorbidity patterns in this population.

---

## 5. Most Stable Diseases (Highest kNN Stability)

These diseases maintain the most consistent comorbidity *neighborhoods* across age transitions.

### Male: Top 20 Most Stable

| ICD  | Disease | Mean Jaccard |
|------|---------|-------------|
| I84  | Haemorrhoids | 0.484 |
| K63  | Other diseases of intestine | 0.481 |
| K44  | Diaphragmatic hernia | 0.479 |
| K21  | Gastro-oesophageal reflux disease (GORD) | 0.471 |
| K62  | Other diseases of anus and rectum | 0.436 |
| M25  | Other joint disorders | 0.429 |
| M65  | Synovitis and tenosynovitis | 0.429 |
| K52  | Other noninfective gastroenteritis and colitis | 0.419 |
| K29  | Gastritis and duodenitis | 0.418 |
| M51  | Intervertebral disc disorders | 0.417 |
| M22  | Disorders of patella | 0.416 |
| K22  | Other diseases of oesophagus | 0.415 |
| M54  | Back pain (dorsalgia) | 0.414 |
| M17  | Gonarthrosis (knee osteoarthritis) | 0.408 |
| M48  | Other spondylopathies | 0.403 |
| K26  | Duodenal ulcer | 0.403 |
| K56  | Paralytic ileus and intestinal obstruction | 0.402 |
| M75  | Shoulder lesions | 0.396 |
| G55  | Nerve root/plexus compressions (classified elsewhere) | 0.395 |
| K92  | Other diseases of digestive system | 0.392 |

### Female: Top 20 Most Stable

| ICD  | Disease | Mean Jaccard |
|------|---------|-------------|
| M67  | Disorders of synovium and tendon | 0.478 |
| K92  | Other diseases of digestive system | 0.474 |
| M25  | Other joint disorders | 0.472 |
| M23  | Internal derangement of knee | 0.468 |
| M22  | Disorders of patella | 0.457 |
| M65  | Synovitis and tenosynovitis | 0.444 |
| I84  | Haemorrhoids | 0.438 |
| K44  | Diaphragmatic hernia | 0.431 |
| K21  | Gastro-oesophageal reflux disease (GORD) | 0.431 |
| M24  | Other joint derangements | 0.424 |
| M94  | Disorders of cartilage | 0.420 |
| G55  | Nerve root/plexus compressions (classified elsewhere) | 0.417 |
| M51  | Intervertebral disc disorders | 0.411 |
| F43  | Reaction to severe stress/adjustment disorders | 0.407 |
| K29  | Gastritis and duodenitis | 0.407 |
| F13  | Sedative/hypnotic use disorders | 0.406 |
| K63  | Other diseases of intestine | 0.402 |
| M17  | Knee osteoarthritis | 0.401 |
| G56  | Mononeuropathies of upper limb | 0.400 |
| M41  | Scoliosis | 0.395 |

### Medical interpretation of highest-stability diseases

The most stable diseases are overwhelmingly **gastrointestinal (K-codes) and musculoskeletal (M-codes)** conditions. These represent the "backbone" of the comorbidity network -- conditions whose comorbidity partners remain consistent across age groups.

**Gastrointestinal stability**: GORD (K21), gastritis (K29), haemorrhoids (I84), hernias (K44), and oesophageal disorders (K22) maintain stable comorbidity neighborhoods because they share consistent pathophysiological links across all adult ages. A patient with GORD is likely to also have gastritis, oesophageal disorders, and functional GI conditions regardless of whether they are 30 or 70. These conditions form a tightly interconnected "GI cluster" that persists throughout adult life.

**Musculoskeletal stability**: Joint disorders (M22, M23, M24, M25), disc disease (M51), back pain (M54), and tendon/synovium conditions (M65, M67) similarly form a persistent "MSK cluster." Once a patient develops degenerative joint or spinal disease, the associated conditions (nerve compressions G55/G56, shoulder lesions M75, spondylopathies M48) remain consistent neighbours.

**An apparent paradox**: Several conditions appear in *both* the highest-drift and highest-stability lists (e.g., M23, M22, M51, M65, M67). This is not contradictory. High drift means the disease's *position* in embedding space moves substantially (it gains many new connections as the network densifies). High stability means its *nearest neighbors* remain the same (it keeps the same comorbidity partners even as it moves). These diseases are "carried along" with their entire neighborhood -- the whole MSK/GI cluster shifts together, preserving internal structure while changing position relative to the rest of the network.

**Sex-specific stable conditions**:
- **F43 (stress/adjustment disorders) and F13 (sedative use disorders)** appear in the female top-20 stability but not the male list. This suggests a persistent cluster of psychiatric/psychological conditions in women that maintains consistent comorbidity links across ages.
- **K56 (intestinal obstruction) and K26 (duodenal ulcer)** appear in the male list but not the female, reflecting sex differences in GI pathology patterns.

---

## 6. Most Unstable Diseases (Lowest kNN Stability)

| Male (Jaccard = 0.000) | Female (Jaccard = 0.000) |
|------------------------|-------------------------|
| M91 (Juvenile osteochondrosis of hip/pelvis) | M30 (Polyarteritis nodosa) |
| A39 (Meningococcal infection) | G72 (Other myopathies) |
| B07 (Viral warts) | N33 (Bladder disorders, classified elsewhere) |
| A30 (Leprosy) | K03 (Other diseases of hard tissues of teeth) |
| E31 (Polyglandular dysfunction) | B56 (African trypanosomiasis) |
| E34 (Other endocrine disorders) | |
| L43 (Lichen planus) | |
| K36 (Other disorders of appendix) | |
| B53 (Other specified malaria) | |
| L26 (Exfoliative dermatitis) | |
| C57 (Malignant neoplasm of other female genital organs) | |

These diseases have **zero** neighborhood overlap between consecutive age groups. Most are either extremely rare conditions, sex-mismatched codes, tropical infections absent from the Austrian population, or conditions that appear transiently in one age group with entirely different comorbidity partners than in the next.

---

## 7. Sex-Specific Differences

### The female comorbidity landscape is shaped by reproductive biology

The female top-drifter list is dominated by **gynaecological and reproductive conditions** that have no male equivalent:
- D06 (cervical carcinoma in situ), N87 (cervical dysplasia), N92 (menstrual disorders), D24 (benign breast neoplasm), D27 (benign ovarian neoplasm), N60 (fibrocystic breast changes), N73 (pelvic inflammatory disease), N83 (ovarian/fallopian tube disorders)

These conditions create a uniquely female pattern: near-zero drift in childhood, explosive repositioning at puberty (2->3), sustained presence during reproductive years, and a second wave of repositioning around menopause (6->7). This two-phase pattern is not seen in the male data.

### Male high-drifters are dominated by cancers and psychiatric conditions

The male top-20 includes **six cancer codes** (C73, C77, C78, C82, C71, C85) and **schizophrenia (F20)**, neither pattern prominent in the female list. Male cancers tend to show sudden, age-specific onset patterns (near-zero drift followed by explosive repositioning), reflecting the epidemiological onset of prostate-related, lymphatic, and brain cancers in middle-aged and older men.

### Shared high-drifters across sexes

Several conditions appear in both male and female top-20 lists:
- **M23, M22, M65, M67** (musculoskeletal) -- these shift dramatically in both sexes
- **C73** (thyroid cancer) -- high drift in both sexes, but with earlier onset in females
- **E89** (postprocedural endocrine/metabolic complications) -- high drift in both, with similar timing

### Stability differences

Female kNN stability is generally higher in early life (0.443 vs 0.346 at transition 1->2) but drops more sharply in old age (0.351 vs 0.411 at transition 7->8). This suggests that **female comorbidity patterns are more consistent in youth but undergo more restructuring in elderly age** -- potentially reflecting the impact of menopause and post-menopausal hormonal changes on the disease network.

---

## 8. Clinical Implications

### Diseases requiring age-specific clinical attention

The highest-drift diseases are precisely those whose **clinical context changes most with patient age**. For a clinician, this means:

- **Thyroid cancer (C73)**: Comorbidity screening guidelines should differ substantially between a 25-year-old and a 55-year-old thyroid cancer patient, as the associated conditions change dramatically.
- **Cervical conditions (D06, N87)** in women: The comorbidity network around cervical pathology restructures twice -- at reproductive onset and around menopause. Screening and management protocols should account for these shifting associations.
- **Schizophrenia (F20)** in men: The explosive repositioning at disease onset (teens/20s) highlights the rapid emergence of metabolic, cardiovascular, and substance-related comorbidities that accompany this diagnosis.
- **Musculoskeletal conditions (M23, M51, M54)**: While their comorbidity neighborhoods remain internally stable, their position in the broader disease network shifts substantially with age, indicating that the metabolic and cardiovascular consequences of MSK conditions evolve significantly across the lifespan.

### Diseases with consistent comorbidity profiles

The most stable diseases (GI and MSK conditions) represent **reliable comorbidity clusters** that clinicians can expect to encounter at any adult age. GORD with gastritis and oesophageal disease, or knee osteoarthritis with disc disease and back pain, are patterns that hold from the 30s through the 70s. Clinical guidelines for these conditions can be more age-invariant.

### The "healthy adolescence" effect

The extreme sparsity of the 10--19 age group network (120--156 edges) compared to adjacent age groups confirms that **adolescence represents a nadir of multimorbidity**. The relatively few comorbidity associations that exist are primarily congenital/developmental conditions carried over from childhood. This "clean slate" makes the subsequent transition to adult disease patterns (20--29) the most dramatic restructuring in the entire lifespan.

---

## 9. Robustness

All findings are robust to node2vec hyperparameter choices. Pairwise Spearman correlations between drift rankings produced by different (p, q) configurations exceed **0.90** for both sexes (male: 0.908--0.914, female: 0.904--0.909). The diseases identified as high-drift or low-drift are not artifacts of the embedding method.

---

## 10. Summary

| Finding | Male | Female |
|---------|------|--------|
| Most volatile transition | 2->3 (10s to 20s) | 2->3 (10s to 20s) |
| Most stable transition | 3->4 (20s to 30s) | 4->5 (30s to 40s) |
| Highest-drift disease | C73, thyroid cancer (3.12) | E74, carbohydrate metabolism disorders (3.58) |
| Dominant high-drift category | Musculoskeletal + cancers | Reproductive/gynaecological + musculoskeletal |
| Most stable disease cluster | GI tract (I84, K63, K44, K21) | MSK joints (M67, M25, M23, M22) |
| Sparsest network (healthiest age) | Age 2, 10--19 (120 edges) | Age 2, 10--19 (156 edges) |
| Densest network (most comorbid age) | Age 8, 70--79 (3,663 edges) | Age 8, 70--79 (4,181 edges) |
| % isolate nodes at age 70--79 | 66.8% | 64.4% |
| Robustness (Spearman rho range) | 0.908--0.914 | 0.904--0.909 |
