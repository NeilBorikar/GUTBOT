import json
import os
import re

# ========== CONFIGURATION ==========
# Diseases folder
FOLDER = r"C:\Users\NEIL\projects\gutbot-medical-data\medical_data\diseases"

# Your diseases data (PASTE JSON OBJECTS HERE)
DISEASES = [
  
    {
        "name": "Alexithymia",
        "category": "Mental Health Disorder",
        "description": "Difficulty in recognizing, processing, and describing one's own emotions.",
        "causes": ["Genetic factors", "Childhood trauma", "Neurological conditions"],
        "related_symptoms": ["emotional numbness", "difficulty identifying feelings", "limited imagination", "interpersonal problems"],
        "severity": "moderate"
    },
    {
        "name": "Stendhal Syndrome",
        "category": "Mental Health Disorder",
        "description": "Psychosomatic disorder causing rapid heartbeat, dizziness, and confusion when exposed to beautiful art.",
        "causes": ["Overwhelming aesthetic experience", "Emotional sensitivity"],
        "related_symptoms": ["panic attacks", "dizziness", "confusion", "rapid heartbeat"],
        "severity": "mild"
    },
    {
        "name": "Capgras Syndrome",
        "category": "Mental Health Disorder",
        "description": "Delusion that a familiar person has been replaced by an identical impostor.",
        "causes": ["Brain injury", "Dementia", "Schizophrenia"],
        "related_symptoms": ["delusional beliefs", "paranoia", "confusion", "agitation"],
        "severity": "severe"
    },
    {
        "name": "Fregoli Syndrome",
        "category": "Mental Health Disorder",
        "description": "Delusion that different people are actually the same person in disguise.",
        "causes": ["Brain damage", "Schizophrenia", "Neurological disorders"],
        "related_symptoms": ["delusions", "paranoia", "confusion", "social withdrawal"],
        "severity": "severe"
    },
    {
        "name": "Cotard's Syndrome",
        "category": "Mental Health Disorder",
        "description": "Delusion that one is dead, does not exist, or has lost organs or blood.",
        "causes": ["Severe depression", "Brain injury", "Neurological conditions"],
        "related_symptoms": ["nihilistic delusions", "depression", "suicidal thoughts", "neglect of self-care"],
        "severity": "severe"
    },
    {
        "name": "Reduplicative Paramnesia",
        "category": "Mental Health Disorder",
        "description": "Delusion that a place or location has been duplicated or relocated.",
        "causes": ["Brain injury", "Stroke", "Dementia"],
        "related_symptoms": ["delusional beliefs", "confusion", "disorientation", "memory problems"],
        "severity": "moderate"
    },
    {
        "name": "Alien Hand Syndrome",
        "category": "Neurological Disorder",
        "description": "Hand moves involuntarily with a sense of foreign control.",
        "causes": ["Brain surgery", "Stroke", "Neurodegenerative diseases"],
        "related_symptoms": ["involuntary movements", "loss of control", "conflict between hands", "frustration"],
        "severity": "moderate"
    },
    {
        "name": "Synesthesia",
        "category": "Neurological Condition",
        "description": "Blending of senses where stimulation of one sense leads to automatic experiences in another.",
        "causes": ["Genetic factors", "Brain wiring differences"],
        "related_symptoms": ["seeing sounds as colors", "tasting words", "colored hearing", "number-color associations"],
        "severity": "mild"
    },
    {
        "name": "Foreign Accent Syndrome",
        "category": "Neurological Disorder",
        "description": "Speech disorder causing sudden change in accent, usually after brain injury.",
        "causes": ["Stroke", "Traumatic brain injury", "Multiple sclerosis"],
        "related_symptoms": ["altered speech patterns", "foreign-sounding accent", "speech difficulties", "frustration"],
        "severity": "moderate"
    },
    {
        "name": "Savant Syndrome",
        "category": "Neurodevelopmental Disorder",
        "description": "Condition where person with significant mental disabilities demonstrates exceptional abilities in specific areas.",
        "causes": ["Autism spectrum disorder", "Brain injury", "Genetic factors"],
        "related_symptoms": ["extraordinary memory", "calendar calculating", "artistic talent", "social difficulties"],
        "severity": "varies"
    },
    {
        "name": "Alice in Wonderland Syndrome",
        "category": "Neurological Disorder",
        "description": "Distortion of perception where objects appear smaller, larger, or distorted.",
        "causes": ["Migraines", "Epilepsy", "Infections"],
        "related_symptoms": ["micropsia", "macropsia", "time distortion", "body image distortion"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Stiff Person Syndrome",
        "category": "Neurological Disorder",
        "description": "Rare autoimmune neurological disorder causing muscle stiffness and spasms.",
        "causes": ["Autoimmune response", "Genetic predisposition"],
        "related_symptoms": ["muscle stiffness", "painful spasms", "anxiety", "mobility problems"],
        "severity": "severe"
    },
    {
        "name": "Kleine-Levin Syndrome",
        "category": "Sleep Disorder",
        "description": "Recurrent episodes of excessive sleep, cognitive disturbances, and behavioral changes.",
        "causes": ["Unknown", "Possible hypothalamic dysfunction"],
        "related_symptoms": ["hypersomnia", "confusion", "megaphagia", "hypersexuality"],
        "severity": "severe"
    },
    {
        "name": "Exploding Head Syndrome",
        "category": "Sleep Disorder",
        "description": "Hearing loud imagined noises or explosion sensations when falling asleep or waking.",
        "causes": ["Stress", "Sleep deprivation", "Minor seizures"],
        "related_symptoms": ["loud noise sensations", "fear", "rapid heartbeat", "sleep disruption"],
        "severity": "mild"
    },
    {
        "name": "Sleep Paralysis",
        "category": "Sleep Disorder",
        "description": "Temporary inability to move or speak while falling asleep or waking up.",
        "causes": ["Sleep deprivation", "Irregular sleep schedule", "Narcolepsy"],
        "related_symptoms": ["paralysis upon waking", "hallucinations", "fear", "breathing difficulties"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Fatal Familial Insomnia",
        "category": "Sleep Disorder",
        "description": "Rare genetic prion disease causing progressively worsening insomnia leading to death.",
        "causes": ["Genetic mutation", "Prion protein misfolding"],
        "related_symptoms": ["progressive insomnia", "autonomic dysfunction", "dementia", "ataxia"],
        "severity": "severe"
    },
    {
        "name": "Stone Man Syndrome",
        "category": "Genetic Disorder",
        "description": "Progressive ossification of soft tissues into bone, severely restricting movement.",
        "causes": ["ACVR1 gene mutation", "Autosomal dominant inheritance"],
        "related_symptoms": ["bone formation in muscles", "joint stiffness", "progressive immobility", "breathing difficulties"],
        "severity": "severe"
    },
    {
        "name": "Harlequin Ichthyosis",
        "category": "Genetic Disorder",
        "description": "Severe genetic skin disorder causing thick, diamond-shaped plates separated by deep cracks.",
        "causes": ["ABCA12 gene mutation", "Autosomal recessive inheritance"],
        "related_symptoms": ["thick skin plates", "deep cracks", "infection risk", "breathing difficulties"],
        "severity": "severe"
    },
    {
        "name": "Epidermolysis Bullosa",
        "category": "Genetic Disorder",
        "description": "Group of disorders causing extremely fragile skin that blisters and tears easily.",
        "causes": ["Genetic mutations affecting skin proteins"],
        "related_symptoms": ["skin blisters", "wounds", "infection risk", "pain"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Xeroderma Pigmentosum",
        "category": "Genetic Disorder",
        "description": "Extreme sensitivity to ultraviolet light leading to skin cancers and neurological problems.",
        "causes": ["DNA repair gene mutations", "Autosomal recessive inheritance"],
        "related_symptoms": ["severe sunburn", "freckling", "skin cancers", "neurological decline"],
        "severity": "severe"
    },
    {
        "name": "Methemoglobinemia",
        "category": "Genetic Disorder",
        "description": "Blood disorder where hemoglobin cannot release oxygen effectively to tissues.",
        "causes": ["Genetic mutations", "Certain medications", "Chemical exposure"],
        "related_symptoms": ["cyanosis", "fatigue", "shortness of breath", "headache"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Porphyria",
        "category": "Genetic Disorder",
        "description": "Group of disorders affecting heme production, causing neurological or skin symptoms.",
        "causes": ["Genetic mutations in heme pathway", "Environmental triggers"],
        "related_symptoms": ["abdominal pain", "neuropathy", "photosensitivity", "mental changes"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Alpha-1 Antitrypsin Deficiency",
        "category": "Genetic Disorder",
        "description": "Genetic disorder causing lung and liver disease due to protein deficiency.",
        "causes": ["SERPINA1 gene mutation", "Autosomal codominant inheritance"],
        "related_symptoms": ["shortness of breath", "wheezing", "cirrhosis", "jaundice"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Wilson's Disease",
        "category": "Genetic Disorder",
        "description": "Copper accumulation in liver, brain, and other vital organs.",
        "causes": ["ATP7B gene mutation", "Autosomal recessive inheritance"],
        "related_symptoms": ["liver disease", "neurological symptoms", "Kayser-Fleischer rings", "psychiatric symptoms"],
        "severity": "severe"
    },
    {
        "name": "Menkes Disease",
        "category": "Genetic Disorder",
        "description": "Copper deficiency disorder causing neurodegeneration and connective tissue problems.",
        "causes": ["ATP7A gene mutation", "X-linked recessive inheritance"],
        "related_symptoms": ["kinky hair", "developmental delay", "hypothermia", "seizures"],
        "severity": "severe"
    },
    {
        "name": "Krabbe Disease",
        "category": "Genetic Disorder",
        "description": "Destructive disorder of the nervous system due to galactocerebrosidase deficiency.",
        "causes": ["GALC gene mutation", "Autosomal recessive inheritance"],
        "related_symptoms": ["irritability", "muscle weakness", "developmental delay", "vision loss"],
        "severity": "severe"
    },
    {
        "name": "Tay-Sachs Disease",
        "category": "Genetic Disorder",
        "description": "Fatal genetic disorder destroying nerve cells in brain and spinal cord.",
        "causes": ["HEXA gene mutation", "Autosomal recessive inheritance"],
        "related_symptoms": ["developmental regression", "cherry-red spot", "seizures", "paralysis"],
        "severity": "severe"
    },
    {
        "name": "Niemann-Pick Disease",
        "category": "Genetic Disorder",
        "description": "Group of severe metabolic disorders where sphingomyelin accumulates in cells.",
        "causes": ["Genetic mutations affecting lipid metabolism"],
        "related_symptoms": ["enlarged liver/spleen", "neurological decline", "feeding difficulties", "developmental delay"],
        "severity": "severe"
    },
    {
        "name": "Gaucher Disease",
        "category": "Genetic Disorder",
        "description": "Lipid storage disorder causing enlarged liver/spleen, bone pain, and neurological problems.",
        "causes": ["GBA gene mutation", "Autosomal recessive inheritance"],
        "related_symptoms": ["enlarged spleen/liver", "bone pain", "anemia", "neurological symptoms"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Fabry Disease",
        "category": "Genetic Disorder",
        "description": "Lipid storage disorder causing pain, kidney failure, heart disease, and stroke.",
        "causes": ["GLA gene mutation", "X-linked inheritance"],
        "related_symptoms": ["burning pain in hands/feet", "angiokeratomas", "kidney problems", "heart disease"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Mucopolysaccharidosis",
        "category": "Genetic Disorder",
        "description": "Group of disorders where complex sugar molecules accumulate in cells.",
        "causes": ["Genetic mutations affecting lysosomal enzymes"],
        "related_symptoms": ["coarse facial features", "joint stiffness", "developmental delay", "organ enlargement"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Osteogenesis Imperfecta",
        "category": "Genetic Disorder",
        "description": "Brittle bone disease causing frequent fractures from minimal trauma.",
        "causes": ["COL1A1/COL1A2 gene mutations", "Autosomal dominant inheritance"],
        "related_symptoms": ["frequent fractures", "blue sclerae", "hearing loss", "short stature"],
        "severity": "mild-to-severe"
    },
    {
        "name": "Achondroplasia",
        "category": "Genetic Disorder",
        "description": "Most common form of dwarfism characterized by short limbs and normal torso.",
        "causes": ["FGFR3 gene mutation", "Autosomal dominant inheritance"],
        "related_symptoms": ["short stature", "disproportionate limbs", "lordosis", "hydrocephalus risk"],
        "severity": "moderate"
    },
    {
        "name": "Cleidocranial Dysplasia",
        "category": "Genetic Disorder",
        "description": "Disorder affecting bone development, particularly skull and collarbones.",
        "causes": ["RUNX2 gene mutation", "Autosomal dominant inheritance"],
        "related_symptoms": ["absent collarbones", "delayed fontanelle closure", "dental abnormalities", "short stature"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Treacher Collins Syndrome",
        "category": "Genetic Disorder",
        "description": "Genetic disorder characterized by craniofacial deformities.",
        "causes": ["TCOF1, POLR1C, or POLR1D gene mutations"],
        "related_symptoms": ["underdeveloped facial bones", "hearing loss", "cleft palate", "eye abnormalities"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Waardenburg Syndrome",
        "category": "Genetic Disorder",
        "description": "Genetic condition affecting pigmentation, hearing, and facial structure.",
        "causes": ["PAX3, MITF, SNAI2, EDN3, EDNRB, or SOX10 gene mutations"],
        "related_symptoms": ["hearing loss", "different colored eyes", "white forelock", "wide-set eyes"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Sturge-Weber Syndrome",
        "category": "Genetic Disorder",
        "description": "Neurological disorder indicated by port-wine stain birthmark and neurological abnormalities.",
        "causes": ["Somatic GNAQ gene mutation", "Not inherited"],
        "related_symptoms": ["facial birthmark", "seizures", "glaucoma", "developmental delays"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Rett Syndrome",
        "category": "Genetic Disorder",
        "description": "Severe neurological disorder affecting mostly girls, causing regression after normal development.",
        "causes": ["MECP2 gene mutation", "X-linked dominant"],
        "related_symptoms": ["loss of purposeful hand skills", "stereotyped hand movements", "loss of speech", "gait abnormalities"],
        "severity": "severe"
    },
    {
        "name": "Angelman Syndrome",
        "category": "Genetic Disorder",
        "description": "Neurogenetic disorder characterized by developmental delay, speech impairment, and happy demeanor.",
        "causes": ["Deletion/mutation of maternal UBE3A gene", "Genetic imprinting"],
        "related_symptoms": ["developmental delay", "lack of speech", "ataxia", "frequent laughter/smiling"],
        "severity": "severe"
    },
    {
        "name": "Prader-Willi Syndrome",
        "category": "Genetic Disorder",
        "description": "Complex genetic disorder causing insatiable appetite, obesity, and cognitive impairment.",
        "causes": ["Deletion/mutation of paternal chromosome 15", "Genetic imprinting"],
        "related_symptoms": ["insatiable hunger", "obesity", "hypogonadism", "cognitive impairment"],
        "severity": "severe"
    },
    {
        "name": "Cornelia de Lange Syndrome",
        "category": "Genetic Disorder",
        "description": "Developmental disorder affecting multiple body systems with distinctive facial features.",
        "causes": ["NIPBL, SMC1A, SMC3, RAD21, or HDAC8 gene mutations"],
        "related_symptoms": ["distinctive facial features", "growth retardation", "intellectual disability", "limb abnormalities"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Smith-Magenis Syndrome",
        "category": "Genetic Disorder",
        "description": "Developmental disorder with distinctive behavioral characteristics and sleep disturbance.",
        "causes": ["Deletion of chromosome 17p11.2", "RAI1 gene mutation"],
        "related_symptoms": ["sleep disturbances", "self-hugging", "behavioral problems", "speech delay"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "22q11.2 Deletion Syndrome",
        "category": "Genetic Disorder",
        "description": "Chromosomal disorder causing heart defects, immune problems, and developmental delays.",
        "causes": ["Deletion of chromosome 22q11.2"],
        "related_symptoms": ["heart defects", "cleft palate", "immune deficiency", "learning disabilities"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Fragile X Syndrome",
        "category": "Genetic Disorder",
        "description": "Genetic condition causing intellectual disability, behavioral and learning challenges.",
        "causes": ["FMR1 gene mutation", "X-linked"],
        "related_symptoms": ["intellectual disability", "social anxiety", "hand-flapping", "long face, large ears"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Kabuki Syndrome",
        "category": "Genetic Disorder",
        "description": "Multiple congenital anomalies with distinctive facial features resembling Kabuki theater makeup.",
        "causes": ["KMT2D or KDM6A gene mutations"],
        "related_symptoms": ["distinctive facial features", "developmental delay", "skeletal abnormalities", "short stature"],
        "severity": "moderate"
    },
    {
        "name": "Noonan Syndrome",
        "category": "Genetic Disorder",
        "description": "Genetic disorder preventing normal development in various parts of the body.",
        "causes": ["PTPN11, SOS1, RAF1, KRAS, NRAS, BRAF, MAP2K1 gene mutations"],
        "related_symptoms": ["facial abnormalities", "heart defects", "short stature", "bleeding disorders"],
        "severity": "moderate"
    },
    {
        "name": "Cardiofaciocutaneous Syndrome",
        "category": "Genetic Disorder",
        "description": "Rare genetic disorder affecting heart, facial features, and skin.",
        "causes": ["BRAF, MAP2K1, MAP2K2, or KRAS gene mutations"],
        "related_symptoms": ["heart defects", "distinctive facial features", "skin abnormalities", "developmental delay"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Costello Syndrome",
        "category": "Genetic Disorder",
        "description": "Rare condition affecting multiple organ systems with increased cancer risk.",
        "causes": ["HRAS gene mutation"],
        "related_symptoms": ["coarse facial features", "heart defects", "developmental delay", "papillomata"],
        "severity": "severe"
    },
    {
        "name": "CHARGE Syndrome",
        "category": "Genetic Disorder",
        "description": "Complex disorder affecting multiple systems with specific pattern of features.",
        "causes": ["CHD7 gene mutation"],
        "related_symptoms": ["coloboma", "heart defects", "choanal atresia", "growth retardation"],
        "severity": "severe"
    },
    {
        "name": "VACTERL Association",
        "category": "Genetic Disorder",
        "description": "Non-random association of birth defects affecting multiple body systems.",
        "causes": ["Unknown", "Possible genetic and environmental factors"],
        "related_symptoms": ["vertebral defects", "anal atresia", "cardiac defects", "renal abnormalities"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Hereditary Spherocytosis",
        "category": "Genetic Disorder",
        "description": "Inherited disorder of red blood cells causing anemia and jaundice.",
        "causes": ["Mutations in genes for RBC membrane proteins"],
        "related_symptoms": ["anemia", "jaundice", "splenomegaly", "gallstones"],
        "severity": "mild-to-severe"
    },
    {
        "name": "Thalassemia",
        "category": "Genetic Disorder",
        "description": "Inherited blood disorder causing abnormal hemoglobin production and anemia.",
        "causes": ["Genetic mutations affecting hemoglobin production"],
        "related_symptoms": ["anemia", "fatigue", "jaundice", "bone deformities"],
        "severity": "mild-to-severe"
    },
    {
        "name": "Hemochromatosis",
        "category": "Genetic Disorder",
        "description": "Iron overload disorder causing organ damage.",
        "causes": ["HFE gene mutation", "Autosomal recessive"],
        "related_symptoms": ["joint pain", "fatigue", "abdominal pain", "bronze skin"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Maple Syrup Urine Disease",
        "category": "Genetic Disorder",
        "description": "Metabolic disorder causing sweet-smelling urine and neurological problems.",
        "causes": ["BCKDHA, BCKDHB, DBT, or DLD gene mutations"],
        "related_symptoms": ["sweet-smelling urine", "poor feeding", "lethargy", "neurological decline"],
        "severity": "severe"
    },
    {
        "name": "Phenylketonuria",
        "category": "Genetic Disorder",
        "description": "Inability to metabolize phenylalanine, leading to intellectual disability if untreated.",
        "causes": ["PAH gene mutation", "Autosomal recessive"],
        "related_symptoms": ["musty body odor", "intellectual disability", "seizures", "behavioral problems"],
        "severity": "severe (if untreated)"
    },
    {
        "name": "Homocystinuria",
        "category": "Genetic Disorder",
        "description": "Metabolic disorder causing multisystemic complications including thrombosis.",
        "causes": ["CBS gene mutation", "Autosomal recessive"],
        "related_symptoms": ["intellectual disability", "eye lens dislocation", "thrombosis", "skeletal abnormalities"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Galactosemia",
        "category": "Genetic Disorder",
        "description": "Inability to metabolize galactose, causing liver and neurological damage.",
        "causes": ["GALT, GALK1, or GALE gene mutations"],
        "related_symptoms": ["jaundice", "vomiting", "failure to thrive", "cataracts"],
        "severity": "severe"
    },
    {
        "name": "Ornithine Transcarbamylase Deficiency",
        "category": "Genetic Disorder",
        "description": "Urea cycle disorder causing hyperammonemia and neurological damage.",
        "causes": ["OTC gene mutation", "X-linked"],
        "related_symptoms": ["lethargy", "vomiting", "hyperammonemia", "coma"],
        "severity": "severe"
    },
    {
        "name": "Lesch-Nyhan Syndrome",
        "category": "Genetic Disorder",
        "description": "Disorder of purine metabolism causing neurological and behavioral abnormalities.",
        "causes": ["HPRT1 gene mutation", "X-linked recessive"],
        "related_symptoms": ["self-mutilation", "gout", "dystonia", "intellectual disability"],
        "severity": "severe"
    },
    {
        "name": "Ataxia Telangiectasia",
        "category": "Genetic Disorder",
        "description": "Progressive neurodegenerative disorder with immune deficiency and cancer predisposition.",
        "causes": ["ATM gene mutation", "Autosomal recessive"],
        "related_symptoms": ["ataxia", "telangiectasias", "immune deficiency", "cancer risk"],
        "severity": "severe"
    },
    {
        "name": "Bloom Syndrome",
        "category": "Genetic Disorder",
        "description": "Chromosomal breakage disorder causing growth deficiency and cancer predisposition.",
        "causes": ["BLM gene mutation", "Autosomal recessive"],
        "related_symptoms": ["short stature", "sun-sensitive rash", "immune deficiency", "high cancer risk"],
        "severity": "severe"
    },
    {
        "name": "Fanconi Anemia",
        "category": "Genetic Disorder",
        "description": "Disorder causing bone marrow failure and cancer predisposition.",
        "causes": ["Mutations in FA pathway genes", "Autosomal recessive"],
        "related_symptoms": ["bone marrow failure", "congenital anomalies", "growth retardation", "cancer risk"],
        "severity": "severe"
    },
    {
        "name": "Dyskeratosis Congenita",
        "category": "Genetic Disorder",
        "description": "Premature aging syndrome affecting rapidly dividing tissues.",
        "causes": ["DKC1, TERC, TERT, TINF2, or other gene mutations"],
        "related_symptoms": ["nail dystrophy", "reticulated skin pigmentation", "oral leukoplakia", "bone marrow failure"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Werner Syndrome",
        "category": "Genetic Disorder",
        "description": "Premature aging disorder beginning in young adulthood.",
        "causes": ["WRN gene mutation", "Autosomal recessive"],
        "related_symptoms": ["premature aging", "cataracts", "skin changes", "cancer risk"],
        "severity": "severe"
    },
    {
        "name": "Hutchinson-Gilford Progeria",
        "category": "Genetic Disorder",
        "description": "Extremely rare rapid aging disorder in children.",
        "causes": ["LMNA gene mutation", "Autosomal dominant"],
        "related_symptoms": ["growth failure", "aged appearance", "alopecia", "cardiovascular disease"],
        "severity": "severe"
    },
    {
        "name": "Cockayne Syndrome",
        "category": "Genetic Disorder",
        "description": "Premature aging disorder with photosensitivity and neurodegeneration.",
        "causes": ["ERCC6 or ERCC8 gene mutations"],
        "related_symptoms": ["growth failure", "photosensitivity", "neurodegeneration", "premature aging"],
        "severity": "severe"
    },
    {
        "name": "Trichothiodystrophy",
        "category": "Genetic Disorder",
        "description": "Disorder with sulfur-deficient brittle hair and multiple systemic abnormalities.",
        "causes": ["ERCC2, ERCC3, or GTF2H5 gene mutations"],
        "related_symptoms": ["brittle hair", "intellectual disability", "photosensitivity", "short stature"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Ectodermal Dysplasia",
        "category": "Genetic Disorder",
        "description": "Group of disorders affecting development of ectodermal tissues.",
        "causes": ["Various gene mutations", "Different inheritance patterns"],
        "related_symptoms": ["sparse hair", "missing teeth", "reduced sweating", "nail abnormalities"],
        "severity": "mild-to-severe"
    },
    {
        "name": "Incontinentia Pigmenti",
        "category": "Genetic Disorder",
        "description": "X-linked dominant disorder affecting skin, hair, teeth, and nervous system.",
        "causes": ["IKBKG gene mutation", "X-linked dominant"],
        "related_symptoms": ["skin blistering", "swirling pigmentation", "dental abnormalities", "neurological problems"],
        "severity": "moderate"
    },
    {
        "name": "Goltz Syndrome",
        "category": "Genetic Disorder",
        "description": "Disorder affecting skin, skeletal system, eyes, and face.",
        "causes": ["PORCN gene mutation", "X-linked dominant"],
        "related_symptoms": ["skin atrophy", "fat herniation", "limb abnormalities", "eye defects"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Moyamoya Disease",
        "category": "Neurological Disorder",
        "description": "Progressive cerebrovascular disorder caused by blocked arteries at base of brain.",
        "causes": ["Genetic factors", "Possible RNF213 gene mutation"],
        "related_symptoms": ["transient ischemic attacks", "stroke", "seizures", "headaches"],
        "severity": "severe"
    },
    {
        "name": "CADASIL",
        "category": "Genetic Disorder",
        "description": "Cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy.",
        "causes": ["NOTCH3 gene mutation", "Autosomal dominant"],
        "related_symptoms": ["migraines with aura", "stroke", "cognitive decline", "mood disorders"],
        "severity": "severe"
    },
    {
        "name": "CARASIL",
        "category": "Genetic Disorder",
        "description": "Cerebral autosomal recessive arteriopathy with subcortical infarcts and leukoencephalopathy.",
        "causes": ["HTRA1 gene mutation", "Autosomal recessive"],
        "related_symptoms": ["stroke", "dementia", "alopecia", "low back pain"],
        "severity": "severe"
    },
    {
        "name": "Huntington's Disease",
        "category": "Genetic Disorder",
        "description": "Progressive neurodegenerative disorder causing uncontrolled movements and cognitive decline.",
        "causes": ["HTT gene mutation", "Autosomal dominant"],
        "related_symptoms": ["chorea", "cognitive decline", "psychiatric symptoms", "difficulty swallowing"],
        "severity": "severe"
    },
    {
        "name": "Spinocerebellar Ataxia",
        "category": "Genetic Disorder",
        "description": "Group of progressive neurodegenerative disorders affecting coordination.",
        "causes": ["Various gene mutations", "Autosomal dominant"],
        "related_symptoms": ["ataxia", "dysarthria", "oculomotor problems", "peripheral neuropathy"],
        "severity": "severe"
    },
    {
        "name": "Friedreich's Ataxia",
        "category": "Genetic Disorder",
        "description": "Progressive nervous system disorder causing movement problems.",
        "causes": ["FXN gene mutation", "Autosomal recessive"],
        "related_symptoms": ["ataxia", "muscle weakness", "speech problems", "heart disease"],
        "severity": "severe"
    },
    {
        "name": "Machado-Joseph Disease",
        "category": "Genetic Disorder",
        "description": "Type of spinocerebellar ataxia with diverse neurological symptoms.",
        "causes": ["ATXN3 gene mutation", "Autosomal dominant"],
        "related_symptoms": ["ataxia", "dystonia", "bulging eyes", "fasciculations"],
        "severity": "severe"
    },
    {
        "name": "Pantothenate Kinase-Associated Neurodegeneration",
        "category": "Genetic Disorder",
        "description": "Neurodegenerative brain iron accumulation disorder.",
        "causes": ["PANK2 gene mutation", "Autosomal recessive"],
        "related_symptoms": ["dystonia", "parkinsonism", "cognitive decline", "retinal degeneration"],
        "severity": "severe"
    },
    {
        "name": "Infantile Neuroaxonal Dystrophy",
        "category": "Genetic Disorder",
        "description": "Neurodegenerative disorder with axonal swelling and neurological decline.",
        "causes": ["PLA2G6 gene mutation", "Autosomal recessive"],
        "related_symptoms": ["developmental regression", "hypotonia", "spasticity", "visual impairment"],
        "severity": "severe"
    },
    {
        "name": "Aicardi Syndrome",
        "category": "Genetic Disorder",
        "description": "Rare disorder characterized by absence of corpus callosum, infantile spasms, and chorioretinal lacunae.",
        "causes": ["Unknown, possibly X-linked dominant"],
        "related_symptoms": ["infantile spasms", "corpus callosum agenesis", "chorioretinal lacunae", "developmental delay"],
        "severity": "severe"
    },
    {
        "name": "Dravet Syndrome",
        "category": "Genetic Disorder",
        "description": "Severe childhood epilepsy with developmental and behavioral problems.",
        "causes": ["SCN1A gene mutation", "Most cases de novo"],
        "related_symptoms": ["prolonged seizures", "developmental delay", "ataxia", "sleep disturbances"],
        "severity": "severe"
    },
    {
        "name": "Lennox-Gastaut Syndrome",
        "category": "Neurological Disorder",
        "description": "Severe form of childhood-onset epilepsy with multiple seizure types.",
        "causes": ["Various brain injuries/malformations", "Genetic factors"],
        "related_symptoms": ["multiple seizure types", "cognitive impairment", "behavioral problems", "slow spike-wave EEG"],
        "severity": "severe"
    },
    {
        "name": "West Syndrome",
        "category": "Neurological Disorder",
        "description": "Infantile epilepsy syndrome with spasms, developmental regression, and hypsarrhythmia.",
        "causes": ["Various brain abnormalities", "Genetic/metabolic disorders"],
        "related_symptoms": ["infantile spasms", "developmental regression", "hypsarrhythmia on EEG", "visual impairment"],
        "severity": "severe"
    },
    {
        "name": "Landau-Kleffner Syndrome",
        "category": "Neurological Disorder",
        "description": "Childhood disorder causing loss of language skills and epilepsy.",
        "causes": ["Unknown", "Possible autoimmune or genetic factors"],
        "related_symptoms": ["acquired aphasia", "epilepsy", "behavioral problems", "sleep-activated EEG abnormalities"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Rasmussen Encephalitis",
        "category": "Neurological Disorder",
        "description": "Rare inflammatory neurological disease causing severe seizures and brain damage.",
        "causes": ["Autoimmune process", "Possible viral trigger"],
        "related_symptoms": ["focal seizures", "progressive hemiparesis", "cognitive decline", "unilateral brain atrophy"],
        "severity": "severe"
    },
    {
        "name": "Anti-NMDA Receptor Encephalitis",
        "category": "Autoimmune Disorder",
        "description": "Autoimmune encephalitis causing psychiatric symptoms, seizures, and movement disorders.",
        "causes": ["Autoantibodies against NMDA receptors", "Often associated with tumors"],
        "related_symptoms": ["psychiatric symptoms", "seizures", "movement disorders", "autonomic instability"],
        "severity": "severe"
    },
    {
        "name": "Hashimoto Encephalopathy",
        "category": "Autoimmune Disorder",
        "description": "Autoimmune encephalopathy associated with Hashimoto's thyroiditis.",
        "causes": ["Autoimmune process", "Thyroid autoimmunity"],
        "related_symptoms": ["cognitive impairment", "seizures", "psychosis", "myoclonus"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Susac Syndrome",
        "category": "Autoimmune Disorder",
        "description": "Autoimmune disease affecting brain, retina, and inner ear.",
        "causes": ["Autoimmune endotheliopathy"],
        "related_symptoms": ["encephalopathy", "branch retinal artery occlusions", "hearing loss", "headache"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Behçet's Disease",
        "category": "Autoimmune Disorder",
        "description": "Vasculitis causing oral/genital ulcers, skin lesions, and eye inflammation.",
        "causes": ["Autoimmune", "Genetic predisposition", "Environmental triggers"],
        "related_symptoms": ["oral ulcers", "genital ulcers", "uveitis", "skin lesions"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Takayasu Arteritis",
        "category": "Autoimmune Disorder",
        "description": "Large vessel vasculitis affecting aorta and its branches.",
        "causes": ["Autoimmune", "Genetic factors"],
        "related_symptoms": ["fever", "fatigue", "claudication", "decreased pulses"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Giant Cell Arteritis",
        "category": "Autoimmune Disorder",
        "description": "Vasculitis of large and medium arteries, often affecting temporal arteries.",
        "causes": ["Autoimmune", "Age-related immune changes"],
        "related_symptoms": ["headache", "jaw claudication", "vision loss", "scalp tenderness"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Polyarteritis Nodosa",
        "category": "Autoimmune Disorder",
        "description": "Systemic necrotizing vasculitis affecting medium-sized arteries.",
        "causes": ["Autoimmune", "Possible hepatitis B association"],
        "related_symptoms": ["fever", "weight loss", "neuropathy", "abdominal pain"],
        "severity": "severe"
    },
    {
        "name": "Granulomatosis with Polyangiitis",
        "category": "Autoimmune Disorder",
        "description": "Vasculitis affecting small to medium vessels with granuloma formation.",
        "causes": ["Autoimmune", "Possible environmental triggers"],
        "related_symptoms": ["sinusitis", "cough", "hematuria", "skin lesions"],
        "severity": "severe"
    },
    {
        "name": "Eosinophilic Granulomatosis with Polyangiitis",
        "category": "Autoimmune Disorder",
        "description": "Vasculitis affecting small to medium vessels with asthma and eosinophilia.",
        "causes": ["Autoimmune"],
        "related_symptoms": ["asthma", "eosinophilia", "neuropathy", "sinusitis"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Microscopic Polyangiitis",
        "category": "Autoimmune Disorder",
        "description": "Necrotizing vasculitis affecting small vessels without granulomas.",
        "causes": ["Autoimmune"],
        "related_symptoms": ["rapidly progressive glomerulonephritis", "hemoptysis", "purpura", "neuropathy"],
        "severity": "severe"
    },
    {
        "name": "Henoch-Schönlein Purpura",
        "category": "Autoimmune Disorder",
        "description": "Vasculitis affecting small vessels with IgA deposition.",
        "causes": ["Autoimmune", "Often follows infection"],
        "related_symptoms": ["palpable purpura", "arthritis", "abdominal pain", "hematuria"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Cryoglobulinemic Vasculitis",
        "category": "Autoimmune Disorder",
        "description": "Vasculitis associated with cryoglobulins in blood.",
        "causes": ["Autoimmune", "Often associated with hepatitis C"],
        "related_symptoms": ["purpura", "arthralgia", "neuropathy", "glomerulonephritis"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Relapsing Polychondritis",
        "category": "Autoimmune Disorder",
        "description": "Systemic disease characterized by recurrent inflammation of cartilage.",
        "causes": ["Autoimmune"],
        "related_symptoms": ["auricular chondritis", "nasal chondritis", "respiratory chondritis", "arthritis"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Cogan Syndrome",
        "category": "Autoimmune Disorder",
        "description": "Rare disorder characterized by ocular inflammation and vestibuloauditory symptoms.",
        "causes": ["Autoimmune"],
        "related_symptoms": ["interstitial keratitis", "hearing loss", "vertigo", "vasculitis"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Sarcoidosis",
        "category": "Autoimmune Disorder",
        "description": "Inflammatory disease characterized by granuloma formation in multiple organs.",
        "causes": ["Autoimmune", "Genetic predisposition", "Environmental factors"],
        "related_symptoms": ["bilateral hilar lymphadenopathy", "erythema nodosum", "uveitis", "skin lesions"],
        "severity": "mild-to-severe"
    },
    {
        "name": "Amyloidosis",
        "category": "Metabolic Disorder",
        "description": "Group of diseases where abnormal proteins build up in tissues and organs.",
        "causes": ["Genetic mutations", "Chronic inflammatory conditions", "Multiple myeloma"],
        "related_symptoms": ["fatigue", "weight loss", "organ dysfunction", "neuropathy"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Familial Mediterranean Fever",
        "category": "Genetic Disorder",
        "description": "Hereditary autoinflammatory disorder causing recurrent fevers and inflammation.",
        "causes": ["MEFV gene mutation", "Autosomal recessive"],
        "related_symptoms": ["recurrent fever", "abdominal pain", "arthritis", "pleuritis"],
        "severity": "moderate"
    },
    {
        "name": "TNF Receptor-Associated Periodic Syndrome",
        "category": "Genetic Disorder",
        "description": "Autosomal dominant autoinflammatory disorder causing recurrent fevers.",
        "causes": ["TNFRSF1A gene mutation", "Autosomal dominant"],
        "related_symptoms": ["recurrent fever", "abdominal pain", "rash", "conjunctivitis"],
        "severity": "moderate"
    },
    {
        "name": "Hyper-IgD Syndrome",
        "category": "Genetic Disorder",
        "description": "Autoinflammatory disorder with recurrent fevers and elevated IgD.",
        "causes": ["MVK gene mutation", "Autosomal recessive"],
        "related_symptoms": ["recurrent fever", "lymphadenopathy", "abdominal pain", "skin rash"],
        "severity": "moderate"
    },
    {
        "name": "Cryopyrin-Associated Periodic Syndromes",
        "category": "Genetic Disorder",
        "description": "Group of autoinflammatory disorders including familial cold autoinflammatory syndrome.",
        "causes": ["NLRP3 gene mutation", "Autosomal dominant"],
        "related_symptoms": ["urticaria-like rash", "fever", "arthritis", "conjunctivitis"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Blau Syndrome",
        "category": "Genetic Disorder",
        "description": "Autosomal dominant granulomatous inflammatory disorder.",
        "causes": ["NOD2 gene mutation", "Autosomal dominant"],
        "related_symptoms": ["granulomatous dermatitis", "arthritis", "uveitis", "campylodactyly"],
        "severity": "moderate"
    },
    {
        "name": "Chronic Recurrent Multifocal Osteomyelitis",
        "category": "Autoinflammatory Disorder",
        "description": "Autoinflammatory bone disorder causing recurrent bone inflammation.",
        "causes": ["Autoinflammatory", "Possible genetic factors"],
        "related_symptoms": ["bone pain", "fever", "swelling", "palmoplantar pustulosis"],
        "severity": "moderate"
    },
    {
        "name": "PAPA Syndrome",
        "category": "Genetic Disorder",
        "description": "Pyogenic arthritis, pyoderma gangrenosum, and acne syndrome.",
        "causes": ["PSTPIP1 gene mutation", "Autosomal dominant"],
        "related_symptoms": ["pyogenic arthritis", "pyoderma gangrenosum", "acne", "cyst formation"],
        "severity": "moderate"
    },
    {
        "name": "Majeed Syndrome",
        "category": "Genetic Disorder",
        "description": "Autosomal recessive disorder with chronic recurrent multifocal osteomyelitis and anemia.",
        "causes": ["LPIN2 gene mutation", "Autosomal recessive"],
        "related_symptoms": ["chronic recurrent multifocal osteomyelitis", "congenital dyserythropoietic anemia", "inflammatory dermatosis"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Deficiency of IL-1 Receptor Antagonist",
        "category": "Genetic Disorder",
        "description": "Autoinflammatory disorder with neonatal onset of pustulosis and osteomyelitis.",
        "causes": ["IL1RN gene mutation", "Autosomal recessive"],
        "related_symptoms": ["neonatal pustulosis", "osteomyelitis", "periostitis", "pulmonary disease"],
        "severity": "severe"
    },
    {
        "name": "SAPHO Syndrome",
        "category": "Autoinflammatory Disorder",
        "description": "Synovitis, acne, pustulosis, hyperostosis, and osteitis syndrome.",
        "causes": ["Autoinflammatory", "Possible infectious trigger"],
        "related_symptoms": ["palmoplantar pustulosis", "sternoclavicular hyperostosis", "arthritis", "acne"],
        "severity": "moderate"
    },
    {
        "name": "Sweet Syndrome",
        "category": "Autoinflammatory Disorder",
        "description": "Disorder characterized by painful skin lesions and fever.",
        "causes": ["Autoinflammatory", "Often associated with malignancies/infections"],
        "related_symptoms": ["painful skin plaques", "fever", "neutrophilia", "arthralgia"],
        "severity": "moderate"
    },
    {
        "name": "Pyoderma Gangrenosum",
        "category": "Autoinflammatory Disorder",
        "description": "Painful ulcerating skin condition often associated with systemic diseases.",
        "causes": ["Autoinflammatory", "Often associated with IBD/arthritis"],
        "related_symptoms": ["painful skin ulcers", "pathergy", "fever", "arthralgia"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Erythema Elevatum Diutinum",
        "category": "Autoinflammatory Disorder",
        "description": "Chronic fibrosing vasculitis with persistent raised skin lesions.",
        "causes": ["Autoimmune vasculitis"],
        "related_symptoms": ["persistent raised skin lesions", "joint pain", "fever", "fatigue"],
        "severity": "moderate"
    },
    {
        "name": "Granuloma Annulare",
        "category": "Autoimmune Disorder",
        "description": "Benign inflammatory skin condition with ring-shaped lesions.",
        "causes": ["Autoimmune", "Possible trigger factors"],
        "related_symptoms": ["ring-shaped skin lesions", "mild itching", "no systemic symptoms"],
        "severity": "mild"
    },
    {
        "name": "Necrobiosis Lipoidica",
        "category": "Autoimmune Disorder",
        "description": "Granulomatous disorder often associated with diabetes mellitus.",
        "causes": ["Autoimmune", "Often associated with diabetes"],
        "related_symptoms": ["shiny plaques on shins", "yellow-brown color", "telangiectasias", "ulceration risk"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Granulomatous Mastitis",
        "category": "Autoimmune Disorder",
        "description": "Inflammatory breast condition mimicking breast cancer.",
        "causes": ["Autoimmune", "Possible infectious triggers"],
        "related_symptoms": ["breast mass", "pain", "skin erythema", "abscess formation"],
        "severity": "moderate"
    },
    {
        "name": "Orbital Pseudotumor",
        "category": "Autoimmune Disorder",
        "description": "Idiopathic inflammatory condition of the orbit.",
        "causes": ["Autoimmune"],
        "related_symptoms": ["proptosis", "pain", "diplopia", "vision loss"],
        "severity": "moderate"
    },
    {
        "name": "Multifocal Fibrosclerosis",
        "category": "Autoimmune Disorder",
        "description": "Systemic fibrosing disorder affecting multiple organs.",
        "causes": ["Autoimmune"],
        "related_symptoms": ["mediastinal fibrosis", "retroperitoneal fibrosis", "sclerosing cholangitis", "Riedel thyroiditis"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Eosinophilic Esophagitis",
        "category": "Autoimmune Disorder",
        "description": "Chronic immune-mediated esophageal disease.",
        "causes": ["Autoimmune", "Food/environmental allergens"],
        "related_symptoms": ["dysphagia", "food impaction", "heartburn", "chest pain"],
        "severity": "moderate"
    },
    {
        "name": "Eosinophilic Gastroenteritis",
        "category": "Autoimmune Disorder",
        "description": "Eosinophilic infiltration of gastrointestinal tract.",
        "causes": ["Autoimmune", "Food allergies"],
        "related_symptoms": ["abdominal pain", "diarrhea", "nausea", "weight loss"],
        "severity": "moderate"
    },
    {
        "name": "Autoimmune Hepatitis",
        "category": "Autoimmune Disorder",
        "description": "Chronic hepatitis with autoimmune features.",
        "causes": ["Autoimmune", "Genetic predisposition"],
        "related_symptoms": ["fatigue", "jaundice", "arthralgia", "elevated liver enzymes"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Primary Biliary Cholangitis",
        "category": "Autoimmune Disorder",
        "description": "Autoimmune liver disease destroying small bile ducts.",
        "causes": ["Autoimmune", "Genetic factors"],
        "related_symptoms": ["fatigue", "pruritus", "jaundice", "xanthelasma"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Primary Sclerosing Cholangitis",
        "category": "Autoimmune Disorder",
        "description": "Chronic cholestatic liver disease with inflammation and fibrosis of bile ducts.",
        "causes": ["Autoimmune", "Strong association with IBD"],
        "related_symptoms": ["fatigue", "pruritus", "jaundice", "abdominal pain"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Autoimmune Pancreatitis",
        "category": "Autoimmune Disorder",
        "description": "Chronic pancreatitis with autoimmune features.",
        "causes": ["Autoimmune"],
        "related_symptoms": ["abdominal pain", "jaundice", "weight loss", "diabetes"],
        "severity": "moderate"
    },
    {
        "name": "IgG4-Related Disease",
        "category": "Autoimmune Disorder",
        "description": "Systemic fibroinflammatory condition with elevated IgG4.",
        "causes": ["Autoimmune"],
        "related_symptoms": ["organ enlargement", "mass lesions", "elevated IgG4", "response to steroids"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Castleman Disease",
        "category": "Lymphoproliferative Disorder",
        "description": "Rare lymphoproliferative disorder with enlarged lymph nodes.",
        "causes": ["Unknown", "Possible HHV-8 association"],
        "related_symptoms": ["lymphadenopathy", "fever", "weight loss", "organomegaly"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Rosai-Dorfman Disease",
        "category": "Histiocytic Disorder",
        "description": "Histiocytic disorder with massive lymphadenopathy.",
        "causes": ["Unknown", "Possible infectious trigger"],
        "related_symptoms": ["massive lymphadenopathy", "fever", "weight loss", "extranodal involvement"],
        "severity": "moderate"
    },
    {
        "name": "Langerhans Cell Histiocytosis",
        "category": "Histiocytic Disorder",
        "description": "Clonal disorder of Langerhans cells with diverse manifestations.",
        "causes": ["Clonal proliferation", "Possible BRAF mutation"],
        "related_symptoms": ["skin rash", "bone lesions", "diabetes insipidus", "organ dysfunction"],
        "severity": "mild-to-severe"
    },
    {
        "name": "Erdheim-Chester Disease",
        "category": "Histiocytic Disorder",
        "description": "Rare non-Langerhans cell histiocytosis with multisystem involvement.",
        "causes": ["Clonal histiocytic proliferation", "Often BRAF V600E mutation"],
        "related_symptoms": ["bone pain", "retroperitoneal fibrosis", "exophthalmos", "cardiac involvement"],
        "severity": "severe"
    },
    {
        "name": "Hemophagocytic Lymphohistiocytosis",
        "category": "Immune Disorder",
        "description": "Life-threatening hyperinflammatory syndrome.",
        "causes": ["Genetic mutations", "Infections", "Malignancies"],
        "related_symptoms": ["fever", "cytopenias", "hepatosplenomegaly", "hemophagocytosis"],
        "severity": "severe"
    },
    {
        "name": "Macrophage Activation Syndrome",
        "category": "Immune Disorder",
        "description": "Severe complication of rheumatic diseases resembling HLH.",
        "causes": ["Hyperinflammation", "Often in systemic JIA"],
        "related_symptoms": ["persistent fever", "pancytopenia", "hepatosplenomegaly", "coagulopathy"],
        "severity": "severe"
    },
    {
        "name": "Periodic Fever, Aphthous Stomatitis, Pharyngitis, Adenitis Syndrome",
        "category": "Autoinflammatory Disorder",
        "description": "Common periodic fever syndrome in children.",
        "causes": ["Autoinflammatory", "Possible genetic factors"],
        "related_symptoms": ["periodic fever", "aphthous ulcers", "pharyngitis", "cervical adenitis"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Cyclic Vomiting Syndrome",
        "category": "Functional Disorder",
        "description": "Disorder characterized by recurrent episodes of severe nausea and vomiting.",
        "causes": ["Mitochondrial dysfunction", "Autonomic nervous system abnormalities"],
        "related_symptoms": ["recurrent vomiting", "nausea", "abdominal pain", "headache"],
        "severity": "moderate"
    },
    {
        "name": "Abdominal Migraine",
        "category": "Functional Disorder",
        "description": "Recurrent episodes of abdominal pain without headache.",
        "causes": ["Unknown", "Possible migraine variant"],
        "related_symptoms": ["recurrent abdominal pain", "nausea", "vomiting", "pallor"],
        "severity": "moderate"
    },
    {
        "name": "Functional Neurological Disorder",
        "category": "Functional Disorder",
        "description": "Condition with neurological symptoms not explained by disease.",
        "causes": ["Psychological factors", "Stress", "Trauma"],
        "related_symptoms": ["weakness", "seizures", "movement disorders", "sensory symptoms"],
        "severity": "moderate"
    },
    {
        "name": "Complex Regional Pain Syndrome",
        "category": "Pain Disorder",
        "description": "Chronic pain condition usually affecting a limb after injury.",
        "causes": ["Nervous system malfunction", "Often follows trauma"],
        "related_symptoms": ["severe pain", "swelling", "skin changes", "temperature abnormalities"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Fibromyalgia",
        "category": "Pain Disorder",
        "description": "Disorder characterized by widespread musculoskeletal pain.",
        "causes": ["Central sensitization", "Genetic factors", "Stress"],
        "related_symptoms": ["widespread pain", "fatigue", "sleep disturbances", "cognitive problems"],
        "severity": "moderate"
    },
    {
        "name": "Myofascial Pain Syndrome",
        "category": "Pain Disorder",
        "description": "Chronic pain disorder affecting muscles and fascia.",
        "causes": ["Muscle overuse", "Injury", "Stress"],
        "related_symptoms": ["muscle pain", "trigger points", "referred pain", "stiffness"],
        "severity": "moderate"
    },
    {
        "name": "Temporomandibular Joint Disorder",
        "category": "Musculoskeletal Disorder",
        "description": "Disorder affecting jaw joint and muscles controlling jaw movement.",
        "causes": ["Jaw injury", "Arthritis", "Bruxism"],
        "related_symptoms": ["jaw pain", "clicking/popping", "headache", "difficulty chewing"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Bruxism",
        "category": "Sleep Disorder",
        "description": "Teeth grinding or clenching, often during sleep.",
        "causes": ["Stress", "Sleep disorders", "Malocclusion"],
        "related_symptoms": ["teeth grinding", "jaw pain", "headache", "tooth damage"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Restless Legs Syndrome",
        "category": "Neurological Disorder",
        "description": "Irresistible urge to move legs, especially at night.",
        "causes": ["Genetic factors", "Iron deficiency", "Pregnancy"],
        "related_symptoms": ["urge to move legs", "uncomfortable sensations", "worsening at rest", "relief with movement"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Periodic Limb Movement Disorder",
        "category": "Sleep Disorder",
        "description": "Involuntary limb movements during sleep.",
        "causes": ["Unknown", "Often associated with RLS"],
        "related_symptoms": ["repetitive limb movements", "sleep disruption", "daytime sleepiness"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Rapid Eye Movement Sleep Behavior Disorder",
        "category": "Sleep Disorder",
        "description": "Sleep disorder where people act out their dreams.",
        "causes": ["Neurodegenerative diseases", "Medications", "Substance withdrawal"],
        "related_symptoms": ["dream enactment", "vocalizations", "complex motor behaviors", "injury risk"],
        "severity": "moderate"
    },
    {
        "name": "Non-24-Hour Sleep-Wake Disorder",
        "category": "Sleep Disorder",
        "description": "Circadian rhythm disorder where sleep-wake cycle is longer than 24 hours.",
        "causes": ["Circadian rhythm dysfunction", "Common in blindness"],
        "related_symptoms": ["shifting sleep times", "insomnia", "excessive sleepiness", "social impairment"],
        "severity": "moderate"
    },
    {
        "name": "Delayed Sleep Phase Disorder",
        "category": "Sleep Disorder",
        "description": "Circadian rhythm disorder with delayed sleep onset.",
        "causes": ["Genetic factors", "Adolescent development"],
        "related_symptoms": ["late sleep onset", "difficulty waking early", "daytime sleepiness", "social impairment"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Advanced Sleep Phase Disorder",
        "category": "Sleep Disorder",
        "description": "Circadian rhythm disorder with early sleep onset and awakening.",
        "causes": ["Genetic factors", "Aging"],
        "related_symptoms": ["early sleep onset", "early morning awakening", "evening sleepiness"],
        "severity": "mild"
    },
    {
        "name": "Irregular Sleep-Wake Rhythm Disorder",
        "category": "Sleep Disorder",
        "description": "Lack of clear circadian rhythm with multiple sleep periods.",
        "causes": ["Neurological conditions", "Dementia", "Brain injury"],
        "related_symptoms": ["multiple sleep periods", "daytime napping", "nighttime wakefulness", "cognitive impairment"],
        "severity": "moderate"
    },
    {
        "name": "Shift Work Disorder",
        "category": "Sleep Disorder",
        "description": "Sleep problems due to working non-traditional hours.",
        "causes": ["Circadian misalignment", "Work schedule"],
        "related_symptoms": ["insomnia", "excessive sleepiness", "reduced alertness", "mood disturbances"],
        "severity": "moderate"
    },
    {
        "name": "Jet Lag Disorder",
        "category": "Sleep Disorder",
        "description": "Temporary sleep problem due to rapid time zone travel.",
        "causes": ["Rapid time zone changes"],
        "related_symptoms": ["insomnia", "daytime fatigue", "impaired functioning", "GI disturbances"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Central Sleep Apnea",
        "category": "Sleep Disorder",
        "description": "Breathing repeatedly stops during sleep due to lack of respiratory effort.",
        "causes": ["Heart failure", "Stroke", "Medications"],
        "related_symptoms": ["breathing pauses", "awakening with shortness of breath", "daytime sleepiness", "morning headaches"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Obstructive Sleep Apnea",
        "category": "Sleep Disorder",
        "description": "Repeated collapse of upper airway during sleep.",
        "causes": ["Anatomic factors", "Obesity", "Alcohol/sedatives"],
        "related_symptoms": ["loud snoring", "breathing pauses", "daytime sleepiness", "morning headache"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Sleep-Related Hypoventilation",
        "category": "Sleep Disorder",
        "description": "Inadequate ventilation during sleep leading to elevated carbon dioxide.",
        "causes": ["Obesity", "Neuromuscular disorders", "Lung diseases"],
        "related_symptoms": ["daytime sleepiness", "morning headaches", "shortness of breath", "cyanosis"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Sleep-Related Hypoxemia",
        "category": "Sleep Disorder",
        "description": "Low oxygen levels during sleep without apnea/hypopnea.",
        "causes": ["Lung diseases", "High altitude", "Heart failure"],
        "related_symptoms": ["daytime fatigue", "morning headaches", "shortness of breath", "cyanosis"],
        "severity": "moderate-to-severe"
    },
    {
        "name": "Catathrenia",
        "category": "Sleep Disorder",
        "description": "Sleep-related groaning during exhalation.",
        "causes": ["Unknown"],
        "related_symptoms": ["groaning during sleep", "social embarrassment", "no daytime sleepiness"],
        "severity": "mild"
    },
    {
        "name": "Sleep-Related Eating Disorder",
        "category": "Sleep Disorder",
        "description": "Eating during sleep with impaired consciousness.",
        "causes": ["Sleepwalking", "Medications", "Other sleep disorders"],
        "related_symptoms": ["nocturnal eating", "impaired consciousness", "strange food combinations", "weight gain"],
        "severity": "moderate"
    },
    {
        "name": "Sexsomnia",
        "category": "Sleep Disorder",
        "description": "Sexual behavior during sleep.",
        "causes": ["Sleepwalking", "Other sleep disorders"],
        "related_symptoms": ["sexual behavior during sleep", "amnesia for events", "social/relationship problems"],
        "severity": "moderate"
    },
    {
        "name": "Sleep Terrors",
        "category": "Sleep Disorder",
        "description": "Episodes of screaming, intense fear during sleep.",
        "causes": ["Sleep deprivation", "Stress", "Fever"],
        "related_symptoms": ["screaming during sleep", "intense fear", "autonomic arousal", "amnesia for events"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Confusional Arousals",
        "category": "Sleep Disorder",
        "description": "Episodes of confusion during or after arousal from sleep.",
        "causes": ["Sleep deprivation", "Other sleep disorders"],
        "related_symptoms": ["confusion upon waking", "disorientation", "slow speech/thinking", "amnesia for events"],
        "severity": "mild"
    },
    {
        "name": "Sleep Enuresis",
        "category": "Sleep Disorder",
        "description": "Bedwetting during sleep beyond expected age.",
        "causes": ["Developmental delay", "Genetic factors", "Sleep apnea"],
        "related_symptoms": ["bedwetting", "social embarrassment", "sleep disruption"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Sleep-Related Hallucinations",
        "category": "Sleep Disorder",
        "description": "Hallucinations occurring at sleep onset or upon awakening.",
        "causes": ["Narcolepsy", "Sleep deprivation", "Medications"],
        "related_symptoms": ["visual/auditory hallucinations", "fear", "disorientation"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Sleep Starts",
        "category": "Sleep Disorder",
        "description": "Sudden brief contractions at sleep onset.",
        "causes": ["Unknown", "Stress", "Sleep deprivation"],
        "related_symptoms": ["sudden jerks at sleep onset", "brief awakening", "no significant impairment"],
        "severity": "mild"
    },
    {
        "name": "Propriospinal Myoclonus at Sleep Onset",
        "category": "Sleep Disorder",
        "description": "Spinal cord-generated jerks at sleep onset.",
        "causes": ["Unknown"],
        "related_symptoms": ["trunk/limb jerks at sleep onset", "insomnia", "frustration"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Sleep-Related Leg Cramps",
        "category": "Sleep Disorder",
        "description": "Painful muscle cramps during sleep.",
        "causes": ["Dehydration", "Electrolyte imbalance", "Medications"],
        "related_symptoms": ["painful leg cramps", "awakening from sleep", "muscle tenderness"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Sleep-Related Bruxism",
        "category": "Sleep Disorder",
        "description": "Teeth grinding during sleep.",
        "causes": ["Stress", "Sleep disorders", "Malocclusion"],
        "related_symptoms": ["teeth grinding", "jaw pain", "tooth damage", "headache"],
        "severity": "mild-to-moderate"
    },
    {
        "name": "Sleep-Related Rhythmic Movement Disorder",
        "category": "Sleep Disorder",
        "description": "Repetitive, stereotyped movements during sleep.",
        "causes": ["Self-soothing behavior", "Developmental"],
        "related_symptoms": ["body rocking", "head banging", "humming", "sleep disruption"],
        "severity": "mild"
    },
    {
        "name": "Benign Sleep Myoclonus of Infancy",
        "category": "Sleep Disorder",
        "description": "Repetitive myoclonic jerks in sleeping infants.",
        "causes": ["Immature nervous system"],
        "related_symptoms": ["repetitive jerks during sleep", "no awakening", "resolves with age"],
        "severity": "mild"
    },
    {
        "name": "Hypnagogic Foot Tremor",
        "category": "Sleep Disorder",
        "description": "Recurrent foot tremor during sleep onset.",
        "causes": ["Unknown"],
        "related_symptoms": ["foot tremor at sleep onset", "no pain", "minimal sleep disruption"],
        "severity": "mild"
    },
    {
        "name": "Alternating Leg Muscle Activation",
        "category": "Sleep Disorder",
        "description": "Brief alternating leg movements during sleep or wakefulness.",
        "causes": ["Unknown"],
        "related_symptoms": ["alternating leg movements", "brief duration", "minimal sleep disruption"],
        "severity": "mild"
    },
    {
        "name": "Excessive Fragmentary Myoclonus",
        "category": "Sleep Disorder",
        "description": "Small muscle twitches during sleep.",
        "causes": ["Unknown"],
        "related_symptoms": ["small muscle twitches", "no significant sleep disruption", "often asymptomatic"],
        "severity": "mild"
    }

]
# ===================================

def create_safe_filename(name: str) -> str:
    """Create a safe filename from disease name"""
    filename = name.lower().replace(" ", "_")
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[^\w\-_]', '_', filename)
    filename = re.sub(r'_+', '_', filename)
    filename = filename.strip('_')
    return filename + ".json"

# Ensure folder exists
os.makedirs(FOLDER, exist_ok=True)

# Get existing disease files
existing_files = set()
if os.path.exists(FOLDER):
    existing_files = {f for f in os.listdir(FOLDER) if f.endswith('.json')}

print(f"📁 Found {len(existing_files)} existing disease files")
print(f"🦠 Ready to add {len(DISEASES)} new diseases")

added = 0
skipped = 0
failed = 0

for disease in DISEASES:
    try:
        # Determine disease name
        disease_name = disease.get(
            "disease_name",
            disease.get(
                "name",
                disease.get(
                    "condition",
                    "unknown"
                )
            )
        )

        if disease_name == "unknown":
            print("⚠️ Skipping disease: No valid name found")
            failed += 1
            continue

        filename = create_safe_filename(disease_name)

        # Avoid overwriting existing files
        if filename in existing_files:
            print(f"⏭️ Skipping: '{disease_name}' (already exists)")
            skipped += 1
            continue

        # Save disease JSON
        filepath = os.path.join(FOLDER, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(disease, f, indent=4, ensure_ascii=False)

        print(f"✅ Added disease: {disease_name}")
        added += 1

    except Exception as e:
        print(f"❌ Failed to save disease: {e}")
        failed += 1

print("\n" + "=" * 50)
print("🎉 DISEASE IMPORT COMPLETE!")
print(f"   ✅ Added: {added}")
print(f"   ⏭️ Skipped: {skipped}")
print(f"   ❌ Failed: {failed}")
print(f"   📁 Total diseases now: {len(existing_files) + added}")
