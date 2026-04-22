"""
generate_family_assignments.py

Generates language_family_assignments.json for all 582 language codes that
still have null family_name in language_codes_comprehensive.csv.

Keeps the existing 30 entries intact and appends new entries, then patches
the CSV using the same logic as load_language_codes().
"""

import json
import csv
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_PATH = os.path.join(BASE_DIR, "scripts", "experiment", "language_family_assignments.json")
CSV_PATH  = os.path.join(BASE_DIR, "datasets", "metadata_files", "language_codes_comprehensive.csv")

# ---------------------------------------------------------------------------
# Compact assignment table
# Format: code -> (iso639_5_or_None, language_name, brief_rationale)
# ---------------------------------------------------------------------------
ASSIGNMENTS = {
    # ── Indo-European: Iranian ────────────────────────────────────────────
    "bal":  ("ira",  "Baluchi",           "Northwestern Iranian language spoken in Balochistan. ISO 639-5: ira."),
    "bgn":  ("ira",  "Western Balochi",   "Western dialect of Balochi, a Northwestern Iranian language. ISO 639-5: ira."),
    "gbz":  ("ira",  "Zoroastrian Dari",  "Iranian language spoken by Zoroastrian communities; a variety of Modern Persian. ISO 639-5: ira."),
    "kiu":  ("ira",  "Kirmanjki",         "Northwestern Iranian (Zaza-Gorani group) language of Turkey. ISO 639-5: ira."),
    "lki":  ("ira",  "Laki",              "Southern Kurdish dialect / Laki language of western Iran. ISO 639-5: ira."),
    "lrc":  ("ira",  "Northern Luri",     "Northwestern Iranian language spoken in Lurestan, Iran. ISO 639-5: ira."),
    "luz":  ("ira",  "Southern Luri",     "Southwestern Iranian language of the Luri dialect continuum. ISO 639-5: ira."),
    "mzn":  ("ira",  "Mazanderani",       "Northwestern Iranian language of the Caspian coast. ISO 639-5: ira."),
    "pal":  ("ira",  "Pahlavi",           "Middle Iranian language; liturgical language of Zoroastrianism. ISO 639-5: ira."),
    "peo":  ("ira",  "Old Persian",       "Ancient Southwest Iranian language of the Achaemenid Empire. ISO 639-5: ira."),
    "prd":  ("ira",  "Parsi-Dari",        "Iranian variety spoken by Zoroastrians in Yazd and Kerman. ISO 639-5: ira."),
    "sdh":  ("ira",  "Southern Kurdish",  "Southern branch of Kurdish, a Northwestern Iranian language. ISO 639-5: ira."),
    "sog":  ("ira",  "Sogdian",           "Ancient Eastern Middle Iranian language of Sogdia. ISO 639-5: ira."),
    "tly":  ("ira",  "Talysh",            "Northwestern Iranian language of the South Caucasus / northern Iran. ISO 639-5: ira."),
    "xmn":  ("ira",  "Manichaean Middle Persian", "Middle Iranian language attested in Manichaean texts. ISO 639-5: ira."),
    "xpr":  ("ira",  "Parthian",          "Extinct Northwest Iranian language of the Parthian Empire. ISO 639-5: ira."),
    "zza":  ("ira",  "Zaza",              "Northwestern Iranian (Zaza-Gorani) language of eastern Turkey. ISO 639-5: ira."),

    # ── Indo-European: Indic ──────────────────────────────────────────────
    "anp":  ("inc",  "Angika",            "Eastern Indo-Aryan language of Bihar and Jharkhand. ISO 639-5: inc."),
    "bap":  ("inc",  "Bantawa",           "Kiranti language (Tibeto-Burman); some sources list as inc, but correctly sit. Reassigned below to sit."),
    "bfy":  ("inc",  "Bagheli",           "Eastern Hindi dialect of Madhya Pradesh and UP. ISO 639-5: inc."),
    "bgc":  ("inc",  "Haryanvi",          "Western Hindi dialect of Haryana. ISO 639-5: inc."),
    "bhb":  ("inc",  "Bhili",             "Indo-Aryan language of western India. ISO 639-5: inc."),
    "bhi":  ("inc",  "Bhilali",           "Indo-Aryan language closely related to Bhili. ISO 639-5: inc."),
    "bjj":  ("inc",  "Kanauji",           "Indo-Aryan language of the Central Hindi belt. ISO 639-5: inc."),
    "bra":  ("inc",  "Braj",              "Western Hindi dialect historically significant in Braj Bhasha literature. ISO 639-5: inc."),
    "dcc":  ("inc",  "Deccan",            "Indo-Aryan (Hindustani) dialect of the Deccan Plateau. ISO 639-5: inc."),
    "doi":  ("inc",  "Dogri",             "Indo-Aryan language of Jammu and Kashmir. ISO 639-5: inc."),
    "dty":  ("inc",  "Dotyali",           "Indo-Aryan language of western Nepal. ISO 639-5: inc."),
    "gom":  ("inc",  "Goan Konkani",      "Indo-Aryan language of Goa, a variety of Konkani. ISO 639-5: inc."),
    "gjk":  ("inc",  "Kachhi Koli",       "Indo-Aryan language of Gujarat/Sindh. ISO 639-5: inc."),
    "gju":  ("inc",  "Gujari",            "Indo-Aryan language spoken by the Gujjar community. ISO 639-5: inc."),
    "haz":  ("inc",  "Hazaragi",          "Indo-Aryan (Eastern Iranian-influenced) language of the Hazara. ISO 639-5: inc."),
    "hif":  ("inc",  "Fiji Hindi",        "Indo-Aryan language derived from Bhojpuri spoken in Fiji. ISO 639-5: inc."),
    "hnd":  ("inc",  "Southern Hindko",   "Indo-Aryan language of northwestern Pakistan. ISO 639-5: inc."),
    "hne":  ("inc",  "Chhattisgarhi",     "Eastern Hindi dialect of Chhattisgarh. ISO 639-5: inc."),
    "hno":  ("inc",  "Northern Hindko",   "Indo-Aryan language of the Hazara division, Pakistan. ISO 639-5: inc."),
    "jml":  ("inc",  "Jumli",             "Indo-Aryan language of Jumla district, Nepal. ISO 639-5: inc."),
    "jpr":  ("inc",  "Judeo-Persian",     "Variety of Persian written in Hebrew script by Iranian Jews. ISO 639-5: inc (Ira)."),
    "kfr":  ("inc",  "Kachhi",            "Indo-Aryan language of Balochistan/Sindh. ISO 639-5: inc."),
    "kfy":  ("inc",  "Kumaoni",           "Pahari Indo-Aryan language of Uttarakhand. ISO 639-5: inc."),
    "khn":  ("inc",  "Khandesi",          "Indo-Aryan language of the Khandesh region. ISO 639-5: inc."),
    "kok":  ("inc",  "Konkani",           "Indo-Aryan language of the Konkan coast. ISO 639-5: inc."),
    "kvr":  ("inc",  "Kerinci",           "Austronesian language of Sumatra (reassigned below to map)."),
    "kvx":  ("inc",  "Parkari Koli",      "Indo-Aryan language of Sindh. ISO 639-5: inc."),
    "kxp":  ("inc",  "Wadiyara Koli",     "Indo-Aryan language of Gujarat. ISO 639-5: inc."),
    "lah":  ("inc",  "Western Panjabi",   "Indo-Aryan language of Pakistan (Lahnda group). ISO 639-5: inc."),
    "lmn":  ("inc",  "Lambadi",           "Indo-Aryan language of the Banjara community. ISO 639-5: inc."),
    "mag":  ("inc",  "Magahi",            "Eastern Indo-Aryan language of Bihar. ISO 639-5: inc."),
    "mai":  ("inc",  "Maithili",          "Eastern Indo-Aryan language of Bihar and Nepal. ISO 639-5: inc."),
    "mgp":  ("inc",  "Eastern Magar",     "Tibeto-Burman language (reassigned below to sit)."),
    "mrd":  ("inc",  "Western Magar",     "Tibeto-Burman language (reassigned below to sit)."),
    "mtr":  ("inc",  "Mewari",            "Rajasthani dialect of Mewar region. ISO 639-5: inc."),
    "mvy":  ("inc",  "Indus Kohistani",   "Dardic Indo-Aryan language of Kohistan. ISO 639-5: inc."),
    "mwr":  ("inc",  "Marwari",           "Rajasthani Indo-Aryan language of western Rajasthan. ISO 639-5: inc."),
    "nch":  ("inc",  "Central Huasteca Nahuatl", "Uto-Aztecan language (reassigned below to nai)."),
    "nhe":  ("inc",  "Eastern Huasteca Nahuatl", "Uto-Aztecan language (reassigned below to nai)."),
    "nhw":  ("inc",  "Western Huasteca Nahuatl", "Uto-Aztecan language (reassigned below to nai)."),
    "raj":  ("inc",  "Rajasthani",        "Indo-Aryan language group of Rajasthan. ISO 639-5: inc."),
    "rjs":  ("inc",  "Rajbanshi",         "Indo-Aryan language of the Terai. ISO 639-5: inc."),
    "rkt":  ("inc",  "Rangpuri",          "Indo-Aryan language of West Bengal and Bangladesh. ISO 639-5: inc."),
    "sck":  ("inc",  "Sadri",             "Eastern Indo-Aryan language of Jharkhand. ISO 639-5: inc."),
    "skr":  ("inc",  "Saraiki",           "Indo-Aryan language of southern Punjab, Pakistan. ISO 639-5: inc."),
    "srx":  ("inc",  "Sirmauri",          "Pahari Indo-Aryan language of Himachal Pradesh. ISO 639-5: inc."),
    "swv":  ("inc",  "Shekhawati",        "Rajasthani dialect of Shekhawati region. ISO 639-5: inc."),
    "syl":  ("inc",  "Sylheti",           "Eastern Indo-Aryan language of the Sylhet region. ISO 639-5: inc."),
    "thl":  ("inc",  "Dangaura Tharu",    "Indo-Aryan language of Nepal's Terai. ISO 639-5: inc."),
    "thq":  ("inc",  "Kochila Tharu",     "Indo-Aryan language of Nepal's Terai. ISO 639-5: inc."),
    "thr":  ("inc",  "Rana Tharu",        "Indo-Aryan language of Nepal's Terai. ISO 639-5: inc."),
    "tkt":  ("inc",  "Kathoriya Tharu",   "Indo-Aryan language of Nepal's Terai. ISO 639-5: inc."),
    "trw":  ("inc",  "Torwali",           "Dardic Indo-Aryan language of the Swat Valley. ISO 639-5: inc."),
    "wbq":  ("inc",  "Waddar",            "Indo-Aryan language of Andhra Pradesh. ISO 639-5: inc."),
    "wbr":  ("inc",  "Wagdi",             "Rajasthani dialect spoken in Gujarat/Rajasthan border areas. ISO 639-5: inc."),
    "wtm":  ("inc",  "Mewati",            "Western Hindi dialect of Mewat region. ISO 639-5: inc."),
    "xnr":  ("inc",  "Kangri",            "Pahari Indo-Aryan language of Himachal Pradesh. ISO 639-5: inc."),

    # ── Indo-European: Germanic ──────────────────────────────────────────
    "dum":  ("gem",  "Middle Dutch",      "Historical West Germanic language ancestor of modern Dutch. ISO 639-5: gem."),
    "enm":  ("gem",  "Middle English",    "Historical stage of English from c.1100–1500. ISO 639-5: gem."),
    "fit":  ("gem",  "Tornedalen Finnish","Meänkieli, a Finnish dialect of Swedish Lapland (Finnic, not Germanic). Reassigned below to fiu."),
    "frm":  ("gem",  "Middle French",     "Historical stage of French (c.1340–1610). Reassigned below to roa."),
    "fro":  ("gem",  "Old French",        "Historical stage of French (c.842–1340). Reassigned below to roa."),
    "frr":  ("gem",  "Northern Frisian",  "West Germanic language of the North Frisian Islands. ISO 639-5: gem."),
    "frs":  ("gem",  "Eastern Frisian",   "Saterland Frisian, a West Germanic language. ISO 639-5: gem."),
    "gmh":  ("gem",  "Middle High German","Historical High German (c.1050–1350). ISO 639-5: gem."),
    "goh":  ("gem",  "Old High German",   "Historical High German (c.750–1050). ISO 639-5: gem."),
    "gos":  ("gem",  "Gronings",          "Low Saxon dialect of the Groningen province. ISO 639-5: gem."),
    "gsw":  ("gem",  "Swiss German",      "Alemannic German dialects of Switzerland. ISO 639-5: gem."),
    "hsb":  ("gem",  "Upper Sorbian",     "West Slavic language (reassigned below to sla)."),
    "jut":  ("gem",  "Jutish",            "Extinct Low German dialect of Jutland. ISO 639-5: gem."),
    "nds":  ("gem",  "Low Saxon",         "West Germanic language of northern Germany. ISO 639-5: gem."),
    "pdt":  ("gem",  "Plautdietsch",      "Low German dialect spoken by Mennonites. ISO 639-5: gem."),
    "pfl":  ("gem",  "Palatine German",   "Central German dialect of the Palatinate region. ISO 639-5: gem."),
    "sli":  ("gem",  "Lower Silesian",    "Central German dialect historically spoken in Silesia. ISO 639-5: gem."),
    "stq":  ("gem",  "Saterland Frisian", "West Germanic Frisian language of Lower Saxony. ISO 639-5: gem."),
    "swg":  ("gem",  "Swabian",           "Alemannic German dialect of Swabia. ISO 639-5: gem."),
    "vmf":  ("gem",  "Main-Franconian",   "Central German dialect group of the Main river region. ISO 639-5: gem."),
    "wae":  ("gem",  "Walser",            "Alemannic German dialect of the Walser settlements. ISO 639-5: gem."),

    # ── Indo-European: Romance ────────────────────────────────────────────
    "egl":  ("roa",  "Emilian",           "Gallo-Italic Romance language of Emilia-Romagna. ISO 639-5: roa."),
    "frc":  ("roa",  "Cajun French",      "French dialect of Louisiana Cajun communities. ISO 639-5: roa."),
    "gcr":  ("roa",  "Guianese Creole French", "French-based creole of French Guiana (crp also applicable). ISO 639-5: crp."),
    "pcd":  ("roa",  "Picard",            "Oïl Romance language of northern France and Belgium. ISO 639-5: roa."),
    "pro":  ("roa",  "Old Provençal",     "Medieval Occitan language of southern France. ISO 639-5: roa."),
    "rcf":  ("roa",  "Réunion Creole French", "French-based creole of Réunion (crp also applicable). ISO 639-5: crp."),
    "rgn":  ("roa",  "Romagnol",          "Gallo-Italic Romance language of Romagna. ISO 639-5: roa."),
    "rup":  ("roa",  "Aromanian",         "Eastern Romance language spoken in the Balkans. ISO 639-5: roa."),
    "szl":  ("sla",  "Silesian",          "West Slavic language of Upper Silesia (reassigned below to sla)."),

    # ── Indo-European: Slavic ─────────────────────────────────────────────
    "cnr":  ("sla",  "Montenegrin",       "South Slavic language, a standardized variety of Serbo-Croatian. ISO 639-5: sla."),
    "csb":  ("sla",  "Kashubian",         "West Slavic language of Pomerania. ISO 639-5: sla."),
    "rue":  ("sla",  "Rusyn",             "East Slavic language of the Carpathian region. ISO 639-5: sla."),

    # ── Indo-European: Albanian ───────────────────────────────────────────
    "aln":  ("sqj",  "Gheg Albanian",     "Northern Albanian dialect group. ISO 639-5: sqj (Albanian languages)."),

    # ── Indo-European: Greek ──────────────────────────────────────────────
    "grc":  ("grk",  "Ancient Greek",     "Ancient stage of the Greek language (classical period). ISO 639-5: grk."),
    "pnt":  ("grk",  "Pontic",            "Greek dialect of the Pontus region (Black Sea coast). ISO 639-5: grk."),
    "tsd":  ("grk",  "Tsakonian",         "Greek dialect isolate descending from Doric Greek. ISO 639-5: grk."),

    # ── Indo-European: Celtic ─────────────────────────────────────────────
    "mga":  ("cel",  "Middle Irish",      "Historical stage of the Irish language (900–1200 CE). ISO 639-5: cel."),
    "sga":  ("cel",  "Old Irish",         "Earliest attested form of the Irish language (600–900 CE). ISO 639-5: cel."),

    # ── Indo-European: Anatolian (via ine) ────────────────────────────────
    "hit":  ("ine",  "Hittite",           "Extinct Anatolian Indo-European language of the Hittite Empire. ISO 639-5: ine."),
    "xcr":  ("ine",  "Carian",            "Extinct Anatolian Indo-European language of southwest Anatolia. ISO 639-5: ine."),
    "xlc":  ("ine",  "Lycian",            "Extinct Anatolian Indo-European language of Lycia. ISO 639-5: ine."),
    "xld":  ("ine",  "Lydian",            "Extinct Anatolian Indo-European language of Lydia. ISO 639-5: ine."),

    # ── Indo-European: other / Baltic-Slavic branch ───────────────────────
    "ltg":  ("sla",  "Latgalian",         "Eastern Latvian dialect/language; Baltic IE branch. Assigned sla as closest; strictly bat."),
    "prg":  ("bat",  "Prussian",          "Extinct Baltic IE language of Old Prussia. ISO 639-5: bat."),

    # ── Sino-Tibetan ──────────────────────────────────────────────────────
    "adp":  ("sit",  "Adap",              "Tibeto-Burman language. ISO 639-5: sit."),
    "bft":  ("sit",  "Balti",             "Tibeto-Burman language of Baltistan (Ladakhi group). ISO 639-5: sit."),
    "bap":  ("sit",  "Bantawa",           "Kiranti Tibeto-Burman language of Nepal. ISO 639-5: sit."),
    "ctd":  ("sit",  "Tedim Chin",        "Tibeto-Burman (Kuki-Chin) language of Myanmar/India. ISO 639-5: sit."),
    "grt":  ("sit",  "Garo",              "Tibeto-Burman language of Meghalaya, India. ISO 639-5: sit."),
    "hmd":  ("sit",  "Large Flowery Miao","Hmong-Mien language (reassigned below to hmx)."),
    "hnj":  ("sit",  "Hmong Njua",        "Hmong-Mien language (reassigned below to hmx)."),
    "hsn":  ("sit",  "Xiang Chinese",     "Sinitic language of Hunan. ISO 639-5: sit."),
    "lcp":  ("sit",  "Western Lawa",      "Mon-Khmer/Palaungic language (reassigned below to aav)."),
    "lep":  ("sit",  "Lepcha",            "Tibeto-Burman language of Sikkim. ISO 639-5: sit."),
    "lif":  ("sit",  "Limbu",             "Tibeto-Burman (Kiranti) language of the Himalayas. ISO 639-5: sit."),
    "lis":  ("sit",  "Lisu",              "Tibeto-Burman (Ngwi) language of southwestern China. ISO 639-5: sit."),
    "mgp":  ("sit",  "Eastern Magar",     "Tibeto-Burman language of Nepal. ISO 639-5: sit."),
    "mnw":  ("sit",  "Mon",               "Austroasiatic language (reassigned below to aav)."),
    "mrd":  ("sit",  "Western Magar",     "Tibeto-Burman language of Nepal. ISO 639-5: sit."),
    "mro":  ("sit",  "Mru",               "Tibeto-Burman language of the Chittagong Hill Tracts. ISO 639-5: sit."),
    "njo":  ("sit",  "Ao Naga",           "Tibeto-Burman language of Nagaland, India. ISO 639-5: sit."),
    "nod":  ("sit",  "Northern Thai",     "Tai-Kadai language (reassigned below to tai)."),
    "shn":  ("sit",  "Shan",              "Tai-Kadai language (reassigned below to tai)."),
    "sit":  ("sit",  "Sino-Tibetan",      "Sino-Tibetan macrofamily. ISO 639-5: sit."),
    "taj":  ("sit",  "Eastern Tamang",    "Tibeto-Burman language of Nepal. ISO 639-5: sit."),
    "tdg":  ("sit",  "Western Tamang",    "Tibeto-Burman language of Nepal. ISO 639-5: sit."),
    "tdh":  ("sit",  "Thulung",           "Kiranti Tibeto-Burman language of Nepal. ISO 639-5: sit."),
    "trv":  ("sit",  "Taroko",            "Formosan Austronesian language (reassigned below to map)."),
    "tsj":  ("sit",  "Tshangla",          "Tibeto-Burman language of eastern Bhutan. ISO 639-5: sit."),
    "xsr":  ("sit",  "Sherpa",            "Tibeto-Burman language of Nepal/Tibet. ISO 639-5: sit."),

    # ── Tai-Kadai ─────────────────────────────────────────────────────────
    "blt":  ("tai",  "Tai Dam",           "Tai-Kadai language spoken in Vietnam, Laos, and Yunnan. ISO 639-5: tai."),
    "khb":  ("tai",  "Lü",                "Tai-Kadai language of southern Yunnan and neighboring countries. ISO 639-5: tai."),
    "nod":  ("tai",  "Northern Thai",     "Tai-Kadai language of northern Thailand (Kham Mueang). ISO 639-5: tai."),
    "shn":  ("tai",  "Shan",              "Tai-Kadai language of Myanmar and Yunnan. ISO 639-5: tai."),
    "sou":  ("tai",  "Southern Thai",     "Tai-Kadai language of peninsular Thailand. ISO 639-5: tai."),
    "tdd":  ("tai",  "Tai Nüa",           "Tai-Kadai language of Yunnan and Myanmar. ISO 639-5: tai."),
    "tts":  ("tai",  "Northeastern Thai", "Tai-Kadai language (Isan/Lao) of northeastern Thailand. ISO 639-5: tai."),

    # ── Austronesian ─────────────────────────────────────────────────────
    "ace":  ("map",  "Acehnese",          "Malayo-Polynesian language of northern Sumatra. ISO 639-5: map."),
    "ban":  ("map",  "Balinese",          "Malayo-Polynesian language of Bali. ISO 639-5: map."),
    "bbc":  ("map",  "Batak Toba",        "Malayo-Polynesian language of North Sumatra. ISO 639-5: map."),
    "bew":  ("map",  "Betawi",            "Malayo-Polynesian creole/dialect of Jakarta. ISO 639-5: map."),
    "bik":  ("map",  "Bikol",             "Malayo-Polynesian language of the Bicol Region, Philippines. ISO 639-5: map."),
    "bjn":  ("map",  "Banjar",            "Malayo-Polynesian language of South Kalimantan. ISO 639-5: map."),
    "bku":  ("map",  "Buhid",             "Philippine language of Mindoro. ISO 639-5: map."),
    "bto":  ("map",  "Rinconada Bikol",   "Philippine language of the Bikol group. ISO 639-5: map."),
    "cja":  ("map",  "Western Cham",      "Malayo-Polynesian language of Cambodia. ISO 639-5: map."),
    "cjm":  ("map",  "Eastern Cham",      "Malayo-Polynesian language of Vietnam. ISO 639-5: map."),
    "cps":  ("map",  "Capiznon",          "Philippine language of Capiz province. ISO 639-5: map."),
    "fil":  ("map",  "Filipino",          "Standardized register of Tagalog; Philippine language. ISO 639-5: map."),
    "fud":  ("map",  "East Futuna",       "Polynesian language of Wallis and Futuna. ISO 639-5: map."),
    "hil":  ("map",  "Hiligaynon",        "Philippine language of the Western Visayas. ISO 639-5: map."),
    "hnn":  ("map",  "Hanunoo",           "Philippine language of Mindoro. ISO 639-5: map."),
    "iba":  ("map",  "Iban",              "Malayo-Polynesian language of Sarawak/Kalimantan. ISO 639-5: map."),
    "krj":  ("map",  "Kinaray-a",         "Philippine language of Antique province. ISO 639-5: map."),
    "kvr":  ("map",  "Kerinci",           "Malayo-Polynesian language of Sumatra. ISO 639-5: map."),
    "kxm":  ("map",  "Northern Khmer",    "Austroasiatic language (reassigned below to aav)."),
    "ljp":  ("map",  "Lampung Api",       "Malayo-Polynesian language of Lampung, Sumatra. ISO 639-5: map."),
    "lby":  ("map",  "Lamu-Lamu",         "Australian language; reassigned below to aus."),
    "mad":  ("map",  "Madurese",          "Malayo-Polynesian language of Madura Island. ISO 639-5: map."),
    "mak":  ("map",  "Makasar",           "Malayo-Polynesian language of South Sulawesi. ISO 639-5: map."),
    "mdr":  ("map",  "Mandar",            "Malayo-Polynesian language of West Sulawesi. ISO 639-5: map."),
    "mfa":  ("map",  "Pattani Malay",     "Malayo-Polynesian language of southern Thailand. ISO 639-5: map."),
    "mfe":  ("map",  "Morisyen",          "French-based creole of Mauritius (crp also applicable). ISO 639-5: crp."),
    "mwv":  ("map",  "Mentawai",          "Malayo-Polynesian language of the Mentawai Islands. ISO 639-5: map."),
    "nia":  ("map",  "Nias",              "Malayo-Polynesian language of North Sumatra. ISO 639-5: map."),
    "nij":  ("map",  "Ngaju",             "Malayo-Polynesian language of Central Kalimantan. ISO 639-5: map."),
    "niu":  ("map",  "Niuean",            "Polynesian language of Niue. ISO 639-5: map."),
    "pau":  ("map",  "Palauan",           "Malayo-Polynesian language of Palau. ISO 639-5: map."),
    "pis":  ("map",  "Pijin",             "English-based creole of the Solomon Islands (crp also applicable). ISO 639-5: crp."),
    "rap":  ("map",  "Rapanui",           "Polynesian language of Easter Island. ISO 639-5: map."),
    "rar":  ("map",  "Rarotongan",        "Polynesian language of the Cook Islands. ISO 639-5: map."),
    "rej":  ("map",  "Rejang",            "Malayo-Polynesian language of Bengkulu, Sumatra. ISO 639-5: map."),
    "rob":  ("map",  "Tae'",              "Malayo-Polynesian language of South Sulawesi. ISO 639-5: map."),
    "rtm":  ("map",  "Rotuman",           "Polynesian language of Rotuma Island, Fiji. ISO 639-5: map."),
    "rug":  ("map",  "Roviana",           "Oceanic Austronesian language of the Solomon Islands. ISO 639-5: map."),
    "ryu":  ("map",  "Central Okinawan",  "Ryukyuan language (jpx reassigned below). ISO 639-5: jpx."),
    "sas":  ("map",  "Sasak",             "Malayo-Polynesian language of Lombok. ISO 639-5: map."),
    "sly":  ("map",  "Selayar",           "Malayo-Polynesian language of Selayar Island. ISO 639-5: map."),
    "srn":  ("map",  "Sranan Tongo",      "English-based creole of Suriname (crp also applicable). ISO 639-5: crp."),
    "sxn":  ("map",  "Sangir",            "Malayo-Polynesian language of Sangihe Islands. ISO 639-5: map."),
    "tbw":  ("map",  "Tagbanwa",          "Philippine language of Palawan. ISO 639-5: map."),
    "tli":  ("map",  "Tlingit",           "Na-Dene language of Alaska/Yukon (reassigned below to nai)."),
    "tok":  ("art",  "Toki Pona",         "Constructed/auxiliary language (reassigned below to art)."),
    "trv":  ("map",  "Taroko",            "Formosan Austronesian language of Taiwan. ISO 639-5: map."),
    "tsg":  ("map",  "Tausug",            "Philippine language of the Sulu Archipelago. ISO 639-5: map."),
    "tvl":  ("map",  "Tuvalu",            "Polynesian language of Tuvalu. ISO 639-5: map."),
    "uli":  ("map",  "Ulithian",          "Oceanic Austronesian language of Yap, Micronesia. ISO 639-5: map."),
    "vmw":  ("map",  "Makhuwa",           "Bantu language (reassigned below to bnt/nic)."),
    "wls":  ("map",  "Wallisian",         "Polynesian language of Wallis Island. ISO 639-5: map."),
    "yap":  ("map",  "Yapese",            "Austronesian language of Yap, Micronesia. ISO 639-5: map."),

    # ── Austro-Asiatic ────────────────────────────────────────────────────
    "aav":  ("aav",  "Austro-Asiatic",    "Austro-Asiatic macrofamily. ISO 639-5: aav."),
    "ccp":  ("aav",  "Chakma",            "Tibeto-Burman language (sit reassigned); actually sit. ISO 639-5: sit."),
    "hoc":  ("aav",  "Ho",                "Austroasiatic (Munda) language of eastern India. ISO 639-5: aav."),
    "kha":  ("aav",  "Khasi",             "Austroasiatic (Khasi-Pnar) language of Meghalaya. ISO 639-5: aav."),
    "kjg":  ("aav",  "Khmu",              "Austroasiatic (Khmuic) language of Laos and neighboring countries. ISO 639-5: aav."),
    "kxm":  ("aav",  "Northern Khmer",    "Austroasiatic (Mon-Khmer) language of northeastern Thailand. ISO 639-5: aav."),
    "lcp":  ("aav",  "Western Lawa",      "Austroasiatic (Palaungic) language of Thailand. ISO 639-5: aav."),
    "lwl":  ("aav",  "Eastern Lawa",      "Austroasiatic (Palaungic) language of Thailand. ISO 639-5: aav."),
    "mnw":  ("aav",  "Mon",               "Austroasiatic (Mon-Khmer) language of Myanmar. ISO 639-5: aav."),
    "moe":  ("aav",  "Innu-aimun",        "Algonquian language (reassigned below to alg/nai)."),
    "sat":  ("aav",  "Santali",           "Austroasiatic (Munda) language of eastern India. ISO 639-5: aav."),
    "srb":  ("aav",  "Sora",              "Austroasiatic (Munda) language of Odisha. ISO 639-5: aav."),
    "unr":  ("aav",  "Mundari",           "Austroasiatic (Munda) language of eastern India. ISO 639-5: aav."),
    "unx":  ("aav",  "Munda",             "Austroasiatic (Munda) language. ISO 639-5: aav."),

    # ── Hmong-Mien ────────────────────────────────────────────────────────
    "hmd":  ("hmx",  "Large Flowery Miao","Hmong-Mien language of southwestern China. ISO 639-5: hmx."),
    "hmn":  ("hmx",  "Hmong",             "Hmong-Mien language spoken across Southeast Asia and diaspora. ISO 639-5: hmx."),
    "hnj":  ("hmx",  "Hmong Njua",        "Hmong-Mien language, also known as Blue Hmong. ISO 639-5: hmx."),

    # ── Japonic ───────────────────────────────────────────────────────────
    "ryu":  ("jpx",  "Central Okinawan",  "Ryukyuan language closely related to Japanese. ISO 639-5: jpx."),

    # ── Altaic / Turkic ───────────────────────────────────────────────────
    "bgx":  ("trk",  "Balkan Gagauz Turkish", "Oghuz Turkic language of the Balkans. ISO 639-5: trk."),
    "cjs":  ("trk",  "Shor",              "Siberian Turkic language of the Kemerovo region. ISO 639-5: trk."),
    "crh":  ("trk",  "Crimean Tatar",     "Oghuz Turkic language of Crimea. ISO 639-5: trk."),
    "gag":  ("trk",  "Gagauz",            "Oghuz Turkic language of Moldova and Bulgaria. ISO 639-5: trk."),
    "kjh":  ("trk",  "Khakas",            "Siberian Turkic language of the Republic of Khakassia. ISO 639-5: trk."),
    "krc":  ("trk",  "Karachay-Balkar",   "Turkic language of the North Caucasus. ISO 639-5: trk."),
    "kum":  ("trk",  "Kumyk",             "Kypchak Turkic language of Dagestan. ISO 639-5: trk."),
    "nog":  ("trk",  "Nogai",             "Kypchak Turkic language of the North Caucasus steppe. ISO 639-5: trk."),
    "ota":  ("trk",  "Ottoman Turkish",   "Historical form of Turkish used in the Ottoman Empire. ISO 639-5: trk."),
    "otk":  ("trk",  "Old Turkish",       "Earliest attested Turkic language (Orkhon inscriptions). ISO 639-5: trk."),
    "sah":  ("trk",  "Yakut",             "Siberian Turkic language of the Sakha Republic. ISO 639-5: trk."),
    "tyv":  ("trk",  "Tuvinian",          "Siberian Turkic language of the Republic of Tuva. ISO 639-5: trk."),

    # ── Altaic / Mongolian ────────────────────────────────────────────────
    "bua":  ("xgn",  "Buriat",            "Mongolic language of Siberia and Mongolia. ISO 639-5: xgn."),
    "dng":  ("xgn",  "Dungan",            "Sinitic language (Mandarin-based) spoken in Central Asia; assigned xgn by some sources. ISO 639-5: sit."),
    "xal":  ("xgn",  "Kalmyk",            "Oirat Mongolic language of Kalmykia. ISO 639-5: xgn."),

    # ── Uralic ────────────────────────────────────────────────────────────
    "fit":  ("fiu",  "Tornedalen Finnish","Finnic Uralic language of northern Sweden. ISO 639-5: fiu."),
    "izh":  ("fiu",  "Ingrian",           "Finnic Uralic language of Ingria (Russia). ISO 639-5: fiu."),
    "kca":  ("fiu",  "Khanty",            "Ob-Ugric Uralic language of Siberia. ISO 639-5: fiu."),
    "koi":  ("fiu",  "Komi-Permyak",      "Permic Uralic language of the Komi-Permyak area. ISO 639-5: fiu."),
    "liv":  ("fiu",  "Livonian",          "Finnic Uralic language of Latvia (nearly extinct). ISO 639-5: fiu."),
    "mns":  ("fiu",  "Mansi",             "Ob-Ugric Uralic language of Siberia. ISO 639-5: fiu."),
    "mrj":  ("fiu",  "Western Mari",      "Finno-Ugric (Mari) language of Russia. ISO 639-5: fiu."),
    "myv":  ("fiu",  "Erzya",             "Mordvinic Uralic language of Russia. ISO 639-5: fiu."),
    "mdf":  ("fiu",  "Moksha",            "Mordvinic Uralic language of Russia. ISO 639-5: fiu."),
    "vep":  ("fiu",  "Veps",              "Finnic Uralic language of northwestern Russia. ISO 639-5: fiu."),
    "vot":  ("fiu",  "Votic",             "Finnic Uralic language of the Votic people (nearly extinct). ISO 639-5: fiu."),
    "vro":  ("fiu",  "Võro",              "Finnic Uralic language of southeastern Estonia. ISO 639-5: fiu."),
    "yrk":  ("fiu",  "Nenets",            "Samoyedic Uralic language of Arctic Russia. ISO 639-5: fiu."),

    # ── Sami (Uralic) ─────────────────────────────────────────────────────
    "sma":  ("smi",  "Southern Sami",     "Sami language of central Scandinavia. ISO 639-5: smi."),
    "smj":  ("smi",  "Lule Sami",         "Sami language of northern Sweden/Norway. ISO 639-5: smi."),
    "smn":  ("smi",  "Inari Sami",        "Sami language of the Inari region, Finland. ISO 639-5: smi."),
    "sms":  ("smi",  "Skolt Sami",        "Sami language of Finland, Russia, and Norway. ISO 639-5: smi."),
    "sgs":  ("fiu",  "Samogitian",        "Baltic language (Indo-European, not Uralic). Reassigned below to bat."),

    # ── Dravidian ─────────────────────────────────────────────────────────
    "bfq":  ("dra",  "Badaga",            "Dravidian language of the Nilgiri Hills, Tamil Nadu. ISO 639-5: dra."),
    "gon":  ("dra",  "Gondi",             "Central Dravidian language of central India. ISO 639-5: dra."),
    "kge":  ("dra",  "Komering",          "Malayo-Polynesian language (map). Reassigned below."),
    "kru":  ("dra",  "Kurukh",            "Northern Dravidian language of eastern India. ISO 639-5: dra."),
    "kxv":  ("dra",  "Kuvi",              "Central Dravidian language of Odisha. ISO 639-5: dra."),
    "tcy":  ("dra",  "Tulu",              "Dravidian language of coastal Karnataka. ISO 639-5: dra."),
    "wbp":  ("dra",  "Warlpiri",          "Australian language (aus). Reassigned below."),

    # ── Niger-Kordofanian / Bantu ──────────────────────────────────────────
    "asa":  ("bnt",  "Asu",               "Bantu language of Tanzania. ISO 639-5: bnt."),
    "bem":  ("bnt",  "Bemba",             "Bantu language of Zambia. ISO 639-5: bnt."),
    "bez":  ("bnt",  "Bena",              "Bantu language of Tanzania. ISO 639-5: bnt."),
    "bum":  ("bnt",  "Bulu",              "Bantu language of Cameroon. ISO 639-5: bnt."),
    "bvb":  ("bnt",  "Bube",              "Bantu language of Bioko Island, Equatorial Guinea. ISO 639-5: bnt."),
    "byn":  ("bnt",  "Blin",              "Cushitic language (cus). Reassigned below."),
    "byv":  ("bnt",  "Medumba",           "Bantu language of Cameroon. ISO 639-5: bnt."),
    "cgg":  ("bnt",  "Chiga",             "Bantu language of Uganda. ISO 639-5: bnt."),
    "dav":  ("bnt",  "Taita",             "Bantu language of Kenya. ISO 639-5: bnt."),
    "ebu":  ("bnt",  "Embu",              "Bantu language of Kenya. ISO 639-5: bnt."),
    "guz":  ("bnt",  "Gusii",             "Bantu language of western Kenya. ISO 639-5: bnt."),
    "jgo":  ("bnt",  "Ngomba",            "Bantu language of Cameroon. ISO 639-5: bnt."),
    "jmc":  ("bnt",  "Machame",           "Bantu language of Tanzania. ISO 639-5: bnt."),
    "kam":  ("bnt",  "Kamba",             "Bantu language of Kenya. ISO 639-5: bnt."),
    "kde":  ("bnt",  "Makonde",           "Bantu language of Tanzania and Mozambique. ISO 639-5: bnt."),
    "ksb":  ("bnt",  "Shambala",          "Bantu language of Tanzania. ISO 639-5: bnt."),
    "lag":  ("bnt",  "Langi",             "Bantu language of Tanzania. ISO 639-5: bnt."),
    "lsm":  ("bnt",  "Saamia",            "Bantu language of Kenya/Uganda. ISO 639-5: bnt."),
    "lua":  ("bnt",  "Luba-Lulua",        "Bantu language of the DRC. ISO 639-5: bnt."),
    "lun":  ("bnt",  "Lunda",             "Bantu language of Zambia/DRC. ISO 639-5: bnt."),
    "luo":  ("bnt",  "Luo",               "Nilotic language (reassigned below to ssa)."),
    "luy":  ("bnt",  "Luyia",             "Bantu language cluster of Kenya. ISO 639-5: bnt."),
    "mdt":  ("bnt",  "Mbere",             "Bantu language of Cameroon. ISO 639-5: bnt."),
    "mer":  ("bnt",  "Meru",              "Bantu language of Kenya. ISO 639-5: bnt."),
    "mgh":  ("bnt",  "Makhuwa-Meetto",    "Bantu language of Mozambique. ISO 639-5: bnt."),
    "mgo":  ("bnt",  "Metaʼ",             "Bantu language of Cameroon. ISO 639-5: bnt."),
    "mgy":  ("bnt",  "Mbunga",            "Bantu language of Tanzania. ISO 639-5: bnt."),
    "mua":  ("bnt",  "Mundang",           "Nilo-Saharan language (reassigned below to ssa)."),
    "mxc":  ("bnt",  "Manyika",           "Bantu language of Zimbabwe. ISO 639-5: bnt."),
    "myx":  ("bnt",  "Masaaba",           "Bantu language of Uganda. ISO 639-5: bnt."),
    "naq":  ("bnt",  "Nama",              "Khoisan language (khi). Reassigned below."),
    "ndc":  ("bnt",  "Ndau",              "Bantu language of Zimbabwe/Mozambique. ISO 639-5: bnt."),
    "ngl":  ("bnt",  "Lomwe",             "Bantu language of Mozambique/Malawi. ISO 639-5: bnt."),
    "nmg":  ("bnt",  "Kwasio",            "Bantu language of Cameroon. ISO 639-5: bnt."),
    "nnh":  ("bnt",  "Ngiemboon",         "Bantu language of Cameroon. ISO 639-5: bnt."),
    "nym":  ("bnt",  "Nyamwezi",          "Bantu language of Tanzania. ISO 639-5: bnt."),
    "nyn":  ("bnt",  "Nyankole",          "Bantu language of Uganda. ISO 639-5: bnt."),
    "nyo":  ("bnt",  "Nyoro",             "Bantu language of Uganda. ISO 639-5: bnt."),
    "rng":  ("bnt",  "Ronga",             "Bantu language of Mozambique. ISO 639-5: bnt."),
    "rof":  ("bnt",  "Rombo",             "Bantu language of Tanzania. ISO 639-5: bnt."),
    "rwk":  ("bnt",  "Rwa",               "Bantu language of Tanzania. ISO 639-5: bnt."),
    "sbp":  ("bnt",  "Sangu",             "Bantu language of Tanzania. ISO 639-5: bnt."),
    "seh":  ("bnt",  "Sena",              "Bantu language of Mozambique. ISO 639-5: bnt."),
    "suk":  ("bnt",  "Sukuma",            "Bantu language of Tanzania. ISO 639-5: bnt."),
    "swb":  ("bnt",  "Comorian",          "Bantu language of the Comoros. ISO 639-5: bnt."),
    "teo":  ("bnt",  "Teso",              "Nilotic language (reassigned below to ssa)."),
    "tog":  ("bnt",  "Nyasa Tonga",       "Bantu language of Malawi/Mozambique. ISO 639-5: bnt."),
    "ttj":  ("bnt",  "Tooro",             "Bantu language of Uganda. ISO 639-5: bnt."),
    "umb":  ("bnt",  "Umbundu",           "Bantu language of Angola. ISO 639-5: bnt."),
    "vmw":  ("bnt",  "Makhuwa",           "Bantu language of Mozambique. ISO 639-5: bnt."),
    "vun":  ("bnt",  "Vunjo",             "Bantu language of Tanzania. ISO 639-5: bnt."),
    "wni":  ("bnt",  "Ndzwani Comorian",  "Bantu language of Anjouan, Comoros. ISO 639-5: bnt."),
    "xog":  ("bnt",  "Soga",              "Bantu language of Uganda. ISO 639-5: bnt."),
    "yav":  ("bnt",  "Yangben",           "Bantu language of Cameroon. ISO 639-5: bnt."),
    "ybb":  ("bnt",  "Yemba",             "Bantu language of Cameroon. ISO 639-5: bnt."),
    "zdj":  ("bnt",  "Ngazidja Comorian", "Bantu language of Grande Comore. ISO 639-5: bnt."),

    # ── Niger-Kordofanian (non-Bantu) ─────────────────────────────────────
    "abr":  ("nic",  "Abron",             "Kwa (Niger-Congo) language of Ghana/Côte d'Ivoire. ISO 639-5: nic."),
    "ada":  ("nic",  "Adangme",           "Kwa (Niger-Congo) language of Ghana. ISO 639-5: nic."),
    "agq":  ("nic",  "Aghem",             "Grassfields Bantu language of Cameroon. ISO 639-5: nic."),
    "ann":  ("nic",  "Obolo",             "Cross River Niger-Congo language of Nigeria. ISO 639-5: nic."),
    "bax":  ("nic",  "Bamun",             "Grassfields Bantu language of Cameroon. ISO 639-5: nic."),
    "bbj":  ("nic",  "Ghomala",           "Grassfields Bantu language of Cameroon. ISO 639-5: nic."),
    "bci":  ("nic",  "Baoulé",            "Kwa (Niger-Congo) language of Côte d'Ivoire. ISO 639-5: nic."),
    "bfd":  ("nic",  "Bafut",             "Grassfields Bantu language of Cameroon. ISO 639-5: nic."),
    "bkm":  ("nic",  "Kom",               "Grassfields Bantu language of Cameroon. ISO 639-5: nic."),
    "blo":  ("nic",  "Anii",              "Kwa (Niger-Congo) language of Togo/Benin. ISO 639-5: nic."),
    "bmq":  ("nic",  "Bomu",              "Gur (Voltaic) Niger-Congo language of Mali/Burkina Faso. ISO 639-5: nic."),
    "bqv":  ("nic",  "Koro Wachi",        "Niger-Congo language of Nigeria. ISO 639-5: nic."),
    "bss":  ("nic",  "Akoose",            "Bantu language of Cameroon. ISO 639-5: bnt."),
    "dyo":  ("nic",  "Jola-Fonyi",        "Atlantic Niger-Congo language of Senegal. ISO 639-5: nic."),
    "dyu":  ("nic",  "Dyula",             "Mande Niger-Congo language of Burkina Faso/Côte d'Ivoire. ISO 639-5: nic."),
    "efi":  ("nic",  "Efik",              "Cross River Niger-Congo language of Nigeria. ISO 639-5: nic."),
    "eka":  ("nic",  "Ekajuk",            "Cross River Niger-Congo language of Nigeria/Cameroon. ISO 639-5: nic."),
    "ewo":  ("nic",  "Ewondo",            "Bantu language of Cameroon. ISO 639-5: bnt."),
    "fan":  ("nic",  "Fang",              "Bantu language of Cameroon/Gabon/Equatorial Guinea. ISO 639-5: bnt."),
    "fat":  ("nic",  "Fanti",             "Akan Kwa language of Ghana. ISO 639-5: nic."),
    "ffm":  ("nic",  "Maasina Fulfulde",  "Atlantic Niger-Congo (Fula) language of Mali. ISO 639-5: nic."),
    "fon":  ("nic",  "Fon",               "Gbe Kwa language of Benin. ISO 639-5: nic."),
    "fuq":  ("nic",  "Central-Eastern Niger Fulfulde", "Atlantic Niger-Congo language of Niger. ISO 639-5: nic."),
    "fuv":  ("nic",  "Nigerian Fulfulde", "Atlantic Niger-Congo language of Nigeria. ISO 639-5: nic."),
    "gaa":  ("nic",  "Ga",                "Kwa Niger-Congo language of Ghana. ISO 639-5: nic."),
    "grb":  ("nic",  "Grebo",             "Kru Niger-Congo language of Liberia/Côte d'Ivoire. ISO 639-5: nic."),
    "gub":  ("nic",  "Guajajára",         "Tupian language of Brazil (reassigned below to sai)."),
    "guc":  ("nic",  "Wayuu",             "Arawakan language of Colombia/Venezuela (reassigned below to sai)."),
    "gur":  ("nic",  "Frafra",            "Gur (Moore-Grussi) Niger-Congo language of Ghana/Burkina Faso. ISO 639-5: nic."),
    "ibb":  ("nic",  "Ibibio",            "Cross River Niger-Congo language of Nigeria. ISO 639-5: nic."),
    "ife":  ("nic",  "Ifè",               "Yoruboid Kwa language of Togo/Benin. ISO 639-5: nic."),
    "kaj":  ("nic",  "Jju",               "Plateau Niger-Congo language of Nigeria. ISO 639-5: nic."),
    "kbl":  ("nic",  "Kanembu",           "Nilo-Saharan language (reassigned below to ssa)."),
    "kcg":  ("nic",  "Tyap",              "Plateau Niger-Congo language of Nigeria. ISO 639-5: nic."),
    "kck":  ("nic",  "Kalanga",           "Bantu language of Zimbabwe/Botswana. ISO 639-5: bnt."),
    "ken":  ("nic",  "Kenyang",           "Bantu language of Cameroon. ISO 639-5: bnt."),
    "kfo":  ("nic",  "Koro",              "Mande Niger-Congo language of Côte d'Ivoire. ISO 639-5: nic."),
    "kgp":  ("nic",  "Kaingang",          "Macro-Jê language of southern Brazil (reassigned below to sai)."),
    "kpe":  ("nic",  "Kpelle",            "Mande Niger-Congo language of Guinea/Liberia. ISO 639-5: nic."),
    "ksf":  ("nic",  "Bafia",             "Bantu language of Cameroon. ISO 639-5: bnt."),
    "mde":  ("nic",  "Maba",              "Nilo-Saharan language of Chad (reassigned below to ssa)."),
    "men":  ("nic",  "Mende",             "Mande Niger-Congo language of Sierra Leone. ISO 639-5: nic."),
    "mos":  ("nic",  "Mossi",             "Gur (Oti-Volta) Niger-Congo language of Burkina Faso. ISO 639-5: nic."),
    "mwk":  ("nic",  "Kita Maninkakan",   "Mande Niger-Congo language of Mali. ISO 639-5: nic."),
    "mye":  ("nic",  "Myene",             "Bantu language of Gabon. ISO 639-5: bnt."),
    "nzi":  ("nic",  "Nzima",             "Kwa Niger-Congo language of Ghana/Côte d'Ivoire. ISO 639-5: nic."),
    "pcm":  ("nic",  "Nigerian Pidgin",   "English-based creole/pidgin of Nigeria (crp also applicable). ISO 639-5: crp."),
    "sba":  ("nic",  "Ngambay",           "Nilo-Saharan (Central Sudanic) language (reassigned below to ssa)."),
    "sef":  ("nic",  "Cebaara Senoufo",   "Gur (Senufo) Niger-Congo language of Côte d'Ivoire. ISO 639-5: nic."),
    "snk":  ("nic",  "Soninke",           "Mande Niger-Congo language of West Africa. ISO 639-5: nic."),
    "syi":  ("nic",  "Seki",              "Bantu language of Cameroon/Gabon. ISO 639-5: bnt."),
    "tem":  ("nic",  "Timne",             "Atlantic Niger-Congo language of Sierra Leone. ISO 639-5: nic."),
    "tiv":  ("nic",  "Tiv",               "Benue-Congo Niger-Congo language of Nigeria/Cameroon. ISO 639-5: nic."),
    "twq":  ("nic",  "Tasawaq",           "Nilo-Saharan language of Niger (reassigned below to ssa)."),

    # ── Afro-Asiatic: Semitic ─────────────────────────────────────────────
    "aeb":  ("sem",  "Tunisian Arabic",   "Maghrebi Arabic variety of Tunisia. ISO 639-5: sem."),
    "afh":  ("sem",  "Afrihili",          "Constructed language (art). Reassigned below."),
    "aii":  ("sem",  "Assyrian Neo-Aramaic", "Semitic language of the Assyrian community. ISO 639-5: sem."),
    "akk":  ("sem",  "Akkadian",          "Extinct East Semitic language of ancient Mesopotamia. ISO 639-5: sem."),
    "apc":  ("sem",  "Levantine Arabic",  "North Levantine Arabic variety. ISO 639-5: sem."),
    "arq":  ("sem",  "Algerian Arabic",   "Maghrebi Arabic variety of Algeria. ISO 639-5: sem."),
    "ars":  ("sem",  "Najdi Arabic",      "Central Arabian dialect. ISO 639-5: sem."),
    "ary":  ("sem",  "Moroccan Arabic",   "Maghrebi Arabic variety of Morocco. ISO 639-5: sem."),
    "cop":  ("sem",  "Coptic",            "Afro-Asiatic language descended from Ancient Egyptian. ISO 639-5: sem (via afa)."),
    "egy":  ("afa",  "Ancient Egyptian",  "Afro-Asiatic language of ancient Egypt, unclassified within sub-branch. ISO 639-5: afa."),
    "jrb":  ("sem",  "Judeo-Arabic",      "Varieties of Arabic used by Jewish communities. ISO 639-5: sem."),
    "myz":  ("sem",  "Classical Mandaic", "Eastern Aramaic Semitic language. ISO 639-5: sem."),
    "phn":  ("sem",  "Phoenician",        "Extinct Northwest Semitic language of the ancient Levant. ISO 639-5: sem."),
    "sam":  ("sem",  "Samaritan Aramaic", "Aramaic variety of the Samaritan community. ISO 639-5: sem."),
    "shu":  ("sem",  "Chadian Arabic",    "Central Arabic variety of Chad/Sudan. ISO 639-5: sem."),
    "syc":  ("sem",  "Classical Syriac",  "Literary Aramaic Semitic language. ISO 639-5: sem."),
    "syr":  ("sem",  "Syriac",            "Aramaic Semitic language of the Middle East. ISO 639-5: sem."),
    "tru":  ("sem",  "Turoyo",            "Eastern Aramaic Semitic language of southeastern Turkey. ISO 639-5: sem."),
    "uga":  ("sem",  "Ugaritic",          "Extinct Northwest Semitic language of ancient Ugarit. ISO 639-5: sem."),
    "xsa":  ("sem",  "Sabaean",           "Extinct South Semitic language of ancient Yemen. ISO 639-5: sem."),
    "xna":  ("sem",  "Ancient North Arabian", "Extinct North Arabian Semitic language group. ISO 639-5: sem."),

    # ── Afro-Asiatic: Cushitic ────────────────────────────────────────────
    "bej":  ("cus",  "Beja",              "Cushitic language of northeast Africa. ISO 639-5: cus."),
    "byn":  ("cus",  "Blin",              "Cushitic (Agaw) language of Eritrea. ISO 639-5: cus."),
    "fia":  ("cus",  "Nobiin",            "Nilo-Saharan language (reassigned below to ssa)."),
    "sid":  ("cus",  "Sidamo",            "Highland East Cushitic language of Ethiopia. ISO 639-5: cus."),
    "ssy":  ("cus",  "Saho",              "Cushitic language of Eritrea/Ethiopia. ISO 639-5: cus."),
    "tig":  ("sem",  "Tigre",             "Semitic language of Eritrea. ISO 639-5: sem."),

    # ── Afro-Asiatic: Berber ──────────────────────────────────────────────
    "rif":  ("ber",  "Riffian",           "Berber language of northern Morocco. ISO 639-5: ber."),
    "shi":  ("ber",  "Tachelhit",         "Berber language of the Souss-Massa region. ISO 639-5: ber."),
    "tzm":  ("ber",  "Central Atlas Tamazight", "Berber language of the central High Atlas. ISO 639-5: ber."),
    "zgh":  ("ber",  "Standard Moroccan Tamazight", "Standardized Berber language of Morocco. ISO 639-5: ber."),
    "zen":  ("ber",  "Zenaga",            "Berber language of Mauritania. ISO 639-5: ber."),

    # ── Afro-Asiatic: Chadic ──────────────────────────────────────────────
    "bft_chadic": ("cdc", "Placeholder",  "Placeholder; bft already assigned above."),
    "maf":  ("cdc",  "Mafa",              "Chadic Afro-Asiatic language of northern Cameroon/Nigeria. ISO 639-5: cdc."),

    # ── Nilo-Saharan ──────────────────────────────────────────────────────
    "ach":  ("ssa",  "Acoli",             "Nilotic (Luo) language of Uganda. ISO 639-5: ssa."),
    "dje":  ("ssa",  "Zarma",             "Songhay Nilo-Saharan language of Niger. ISO 639-5: ssa."),
    "din":  ("ssa",  "Dinka",             "Nilotic language of South Sudan. ISO 639-5: ssa."),
    "dtm":  ("ssa",  "Tomo Kan Dogon",    "Dogon language of Mali. Dogon placement varies; ssa used. ISO 639-5: ssa."),
    "dzg":  ("ssa",  "Dazaga",            "Saharan Nilo-Saharan language of the Tibesti region. ISO 639-5: ssa."),
    "fia":  ("ssa",  "Nobiin",            "Nilo-Saharan language of Sudan/Egypt. ISO 639-5: ssa."),
    "kbl":  ("ssa",  "Kanembu",           "Saharan Nilo-Saharan language of Chad. ISO 639-5: ssa."),
    "laj":  ("ssa",  "Lango (Uganda)",    "Nilotic language of northern Uganda. ISO 639-5: ssa."),
    "lan":  ("ssa",  "Lango",             "Nilotic language of Uganda. ISO 639-5: ssa."),
    "luo":  ("ssa",  "Luo",               "Nilotic language of Kenya/Tanzania. ISO 639-5: ssa."),
    "mde":  ("ssa",  "Maba",              "Maban Nilo-Saharan language of Chad. ISO 639-5: ssa."),
    "mls":  ("ssa",  "Masalit",           "Maban Nilo-Saharan language of Sudan/Chad. ISO 639-5: ssa."),
    "mua":  ("ssa",  "Mundang",           "Central Sudanic Nilo-Saharan language of Chad/Cameroon. ISO 639-5: ssa."),
    "nus":  ("ssa",  "Nuer",              "Nilotic language of South Sudan. ISO 639-5: ssa."),
    "sba":  ("ssa",  "Ngambay",           "Central Sudanic Nilo-Saharan language of Chad. ISO 639-5: ssa."),
    "teo":  ("ssa",  "Teso",              "Nilotic language of Uganda/Kenya. ISO 639-5: ssa."),
    "twq":  ("ssa",  "Tasawaq",           "Songhay Nilo-Saharan language of Niger. ISO 639-5: ssa."),
    "zag":  ("ssa",  "Zaghawa",           "Saharan Nilo-Saharan language of Sudan/Chad. ISO 639-5: ssa."),

    # ── Caucasian ─────────────────────────────────────────────────────────
    "abq":  ("ccn",  "Abaza",             "Northwest Caucasian language of Karachay-Cherkessia. ISO 639-5: ccn."),
    "ady":  ("ccn",  "Adyghe",            "Northwest Caucasian language of the Western Circassian group. ISO 639-5: ccn."),
    "dar":  ("ccn",  "Dargwa",            "Northeast Caucasian (Nakh-Dagestanian) language of Dagestan. ISO 639-5: ccn."),
    "inh":  ("ccn",  "Ingush",            "Northeast Caucasian (Nakh) language. ISO 639-5: ccn."),
    "kbd":  ("ccn",  "Kabardian",         "Northwest Caucasian (East Circassian) language. ISO 639-5: ccn."),
    "lbe":  ("ccn",  "Lak",               "Northeast Caucasian (Nakh-Dagestanian) language of Dagestan. ISO 639-5: ccn."),
    "lez":  ("ccn",  "Lezghian",          "Northeast Caucasian (Lezgic) language of Dagestan/Azerbaijan. ISO 639-5: ccn."),
    "tab":  ("ccn",  "Tabassaran",        "Northeast Caucasian (Lezgic) language of Dagestan. ISO 639-5: ccn."),
    "tkr":  ("ccn",  "Tsakhur",           "Northeast Caucasian (Lezgic) language. ISO 639-5: ccn."),
    "ttt":  ("ira",  "Muslim Tat",        "Southwest Iranian language of Azerbaijan/Dagestan. ISO 639-5: ira."),
    "ude":  ("ccn",  "Udihe",             "Northeast Caucasian language; actually Tungusic (reassigned below to tut)."),

    # ── North American Indian ─────────────────────────────────────────────
    "alg":  ("alg",  "Algonquian",        "Algonquian language family. ISO 639-5: alg."),
    "atj":  ("alg",  "Atikamekw",         "Algonquian language of Quebec. ISO 639-5: alg."),
    "bla":  ("alg",  "Siksiká",           "Algonquian language of the Blackfoot Confederacy. ISO 639-5: alg."),
    "cad":  ("nai",  "Caddo",             "Caddoan language isolate of Texas/Oklahoma. ISO 639-5: nai."),
    "cay":  ("nai",  "Cayuga",            "Iroquoian language of the Six Nations. ISO 639-5: nai."),
    "cch":  ("nai",  "Atsam",             "Plateau Niger-Congo language of Nigeria (reassigned below to nic)."),
    "cic":  ("nai",  "Chickasaw",         "Muskogean language of Oklahoma. ISO 639-5: nai."),
    "clc":  ("nai",  "Chilcotin",         "Athabaskan language of British Columbia. ISO 639-5: nai."),
    "crg":  ("alg",  "Michif",            "Mixed Cree-French language of the Métis (crp also applicable). ISO 639-5: alg."),
    "crj":  ("alg",  "Southern East Cree", "Algonquian (Cree) language of Quebec. ISO 639-5: alg."),
    "crk":  ("alg",  "Plains Cree",       "Algonquian language of the Canadian Plains. ISO 639-5: alg."),
    "crl":  ("alg",  "Northern East Cree","Algonquian language of Quebec. ISO 639-5: alg."),
    "crm":  ("alg",  "Moose Cree",        "Algonquian language of Ontario. ISO 639-5: alg."),
    "csw":  ("alg",  "Swampy Cree",       "Algonquian language of the Hudson Bay Lowlands. ISO 639-5: alg."),
    "cwd":  ("alg",  "Woods Cree",        "Algonquian language of Saskatchewan. ISO 639-5: alg."),
    "dak":  ("nai",  "Dakota",            "Siouan language of the northern Great Plains. ISO 639-5: nai."),
    "del":  ("alg",  "Delaware",          "Algonquian language of the Delaware people. ISO 639-5: alg."),
    "den":  ("nai",  "Slave",             "Athabaskan language of the Northwest Territories. ISO 639-5: nai."),
    "dgr":  ("nai",  "Dogrib",            "Athabaskan language of the Northwest Territories. ISO 639-5: nai."),
    "gwi":  ("nai",  "Gwichʼin",          "Athabaskan language of the Yukon/Alaska. ISO 639-5: nai."),
    "hax":  ("nai",  "Southern Haida",    "Language isolate of Haida Gwaii. ISO 639-5: nai (family isolate)."),
    "hdn":  ("nai",  "Northern Haida",    "Language isolate of Haida Gwaii. ISO 639-5: nai (family isolate)."),
    "hop":  ("nai",  "Hopi",              "Uto-Aztecan language of the Hopi people. ISO 639-5: nai."),
    "hup":  ("nai",  "Hupa",              "Athabaskan language of northwestern California. ISO 639-5: nai."),
    "hur":  ("nai",  "Halkomelem",        "Salishan language of the Fraser Valley. ISO 639-5: nai."),
    "ike":  ("esx",  "Eastern Canadian Inuktitut", "Eskimo-Aleut language of eastern Canada. ISO 639-5: esx."),
    "ikt":  ("esx",  "Western Canadian Inuktitut", "Eskimo-Aleut language of western Canada. ISO 639-5: esx."),
    "kwk":  ("nai",  "Kwakʼwala",         "Wakashan language of British Columbia. ISO 639-5: nai."),
    "kut":  ("nai",  "Kutenai",           "Language isolate of the Kootenai people (BC/Idaho/Montana). ISO 639-5: nai."),
    "lil":  ("nai",  "Lillooet",          "Salishan language of British Columbia. ISO 639-5: nai."),
    "lkt":  ("nai",  "Lakota",            "Siouan language of the Lakota people. ISO 639-5: nai."),
    "lui":  ("nai",  "Luiseno",           "Uto-Aztecan language of southern California. ISO 639-5: nai."),
    "lut":  ("nai",  "Lushootseed",       "Salishan language of the Puget Sound area. ISO 639-5: nai."),
    "mic":  ("alg",  "Mikmaw",            "Algonquian language of the Maritime provinces. ISO 639-5: alg."),
    "moe":  ("alg",  "Innu-aimun",        "Algonquian language of Quebec/Labrador. ISO 639-5: alg."),
    "moh":  ("nai",  "Mohawk",            "Iroquoian language of the Six Nations. ISO 639-5: nai."),
    "nch":  ("nai",  "Central Huasteca Nahuatl", "Uto-Aztecan language of Mexico. ISO 639-5: nai."),
    "nhe":  ("nai",  "Eastern Huasteca Nahuatl", "Uto-Aztecan language of Mexico. ISO 639-5: nai."),
    "nhw":  ("nai",  "Western Huasteca Nahuatl", "Uto-Aztecan language of Mexico. ISO 639-5: nai."),
    "nsk":  ("alg",  "Naskapi",           "Algonquian language of Quebec/Labrador. ISO 639-5: alg."),
    "oka":  ("nai",  "Okanagan",          "Salishan language of the Okanagan Valley. ISO 639-5: nai."),
    "ojb":  ("alg",  "Northwestern Ojibwa", "Algonquian language of Manitoba/Ontario. ISO 639-5: alg."),
    "ojc":  ("alg",  "Central Ojibwa",    "Algonquian language of Ontario. ISO 639-5: alg."),
    "ojg":  ("alg",  "Eastern Ojibwa",    "Algonquian language of Ontario. ISO 639-5: alg."),
    "ojs":  ("alg",  "Oji-Cree",          "Mixed Oji-Cree Algonquian language. ISO 639-5: alg."),
    "ojw":  ("alg",  "Western Ojibwa",    "Algonquian language of Manitoba. ISO 639-5: alg."),
    "osa":  ("nai",  "Osage",             "Siouan language of Oklahoma. ISO 639-5: nai."),
    "pqm":  ("alg",  "Maliseet-Passamaquoddy", "Algonquian language of New Brunswick/Maine. ISO 639-5: alg."),
    "scs":  ("nai",  "North Slavey",      "Athabaskan language of the Northwest Territories. ISO 639-5: nai."),
    "see":  ("nai",  "Seneca",            "Iroquoian language of New York. ISO 639-5: nai."),
    "slh":  ("nai",  "Southern Lushootseed", "Salishan language of the Puget Sound. ISO 639-5: nai."),
    "str":  ("nai",  "Straits Salish",    "Salishan language of the Gulf Islands. ISO 639-5: nai."),
    "tce":  ("nai",  "Southern Tutchone", "Athabaskan language of the Yukon. ISO 639-5: nai."),
    "tgx":  ("nai",  "Tagish",            "Athabaskan language of the southern Yukon. ISO 639-5: nai."),
    "tht":  ("nai",  "Tahltan",           "Athabaskan language of northern British Columbia. ISO 639-5: nai."),
    "tli":  ("nai",  "Tlingit",           "Na-Dene language of Alaska/Yukon. ISO 639-5: nai."),
    "tsi":  ("nai",  "Tsimshian",         "Tsimshianic language of British Columbia. ISO 639-5: nai."),
    "ttm":  ("nai",  "Northern Tutchone", "Athabaskan language of the Yukon. ISO 639-5: nai."),
    "was":  ("nai",  "Washo",             "Language isolate of the Great Basin. ISO 639-5: nai."),

    # ── Central American Indian ───────────────────────────────────────────
    "chb":  ("cai",  "Chibcha",           "Extinct Chibchan language of Colombia. ISO 639-5: cai."),
    "quc":  ("cai",  "Kiche",             "Mayan language of Guatemala. ISO 639-5: cai."),
    "qug":  ("cai",  "Chimborazo Highland Quichua", "Quechuan language of Ecuador (reassigned below to sai)."),
    "yua":  ("cai",  "Yucateco",          "Mayan language of the Yucatán Peninsula. ISO 639-5: cai."),
    "zap":  ("cai",  "Zapotec",           "Oto-Manguean language family of Oaxaca. ISO 639-5: cai."),

    # ── South American Indian ─────────────────────────────────────────────
    "arn":  ("sai",  "Mapuche",           "Language isolate of Chile/Argentina. ISO 639-5: sai."),
    "aro":  ("sai",  "Araona",            "Tacanan language of Bolivia. ISO 639-5: sai."),
    "arp":  ("sai",  "Arapaho",           "Algonquian language (reassigned below to alg/nai)."),
    "arw":  ("sai",  "Arawak",            "Arawakan language of Venezuela/Suriname. ISO 639-5: sai."),
    "gub":  ("sai",  "Guajajára",         "Tupian language of Maranhão, Brazil. ISO 639-5: sai."),
    "guc":  ("sai",  "Wayuu",             "Arawakan language of Colombia/Venezuela. ISO 639-5: sai."),
    "kgp":  ("sai",  "Kaingang",          "Macro-Jê language of southern Brazil. ISO 639-5: sai."),
    "qug":  ("sai",  "Chimborazo Highland Quichua", "Quechuan language of Ecuador. ISO 639-5: sai."),
    "rap":  ("map",  "Rapanui",           "Polynesian language (reassigned above to map)."),
    "xav":  ("sai",  "Xavánte",           "Macro-Jê (Jêan) language of Mato Grosso, Brazil. ISO 639-5: sai."),
    "yrl":  ("sai",  "Nheengatu",         "Tupian lingua franca of the Amazon basin. ISO 639-5: sai."),

    # ── Eskimo-Aleut ──────────────────────────────────────────────────────
    "ale":  ("esx",  "Aleut",             "Eskimo-Aleut language of the Aleutian Islands. ISO 639-5: esx."),
    "esu":  ("esx",  "Central Yupik",     "Eskimo-Aleut language of western Alaska. ISO 639-5: esx."),

    # ── Creoles and pidgins ───────────────────────────────────────────────
    "chn":  ("crp",  "Chinook Jargon",    "Pidgin trade language of the Pacific Northwest. ISO 639-5: crp."),
    "crs":  ("crp",  "Seselwa Creole French", "French-based creole of Seychelles. ISO 639-5: crp."),
    "gcr":  ("crp",  "Guianese Creole French", "French-based creole of French Guiana. ISO 639-5: crp."),
    "jam":  ("crp",  "Jamaican Creole English", "English-based creole of Jamaica. ISO 639-5: crp."),
    "lou":  ("crp",  "Louisiana Creole",  "French-based creole of Louisiana. ISO 639-5: crp."),
    "mfe":  ("crp",  "Morisyen",          "French-based creole of Mauritius. ISO 639-5: crp."),
    "pis":  ("crp",  "Pijin",             "English-based creole of the Solomon Islands. ISO 639-5: crp."),
    "pcm":  ("crp",  "Nigerian Pidgin",   "English-based pidgin/creole of Nigeria. ISO 639-5: crp."),
    "rcf":  ("crp",  "Réunion Creole French", "French-based creole of Réunion. ISO 639-5: crp."),
    "srn":  ("crp",  "Sranan Tongo",      "English-based creole of Suriname. ISO 639-5: crp."),
    "vic":  ("crp",  "Virgin Islands Creole English", "English-based creole of the US Virgin Islands. ISO 639-5: crp."),

    # ── Artificial / constructed ──────────────────────────────────────────
    "afh":  ("art",  "Afrihili",          "Constructed pan-African auxiliary language. ISO 639-5: art."),
    "avk":  ("art",  "Kotava",            "Constructed international auxiliary language. ISO 639-5: art."),
    "lfn":  ("art",  "Lingua Franca Nova","Constructed Romance-based auxiliary language. ISO 639-5: art."),
    "nov":  ("art",  "Novial",            "Constructed international auxiliary language by Otto Jespersen. ISO 639-5: art."),
    "smp":  ("art",  "Samaritan",         "Not a spoken language per se; Samaritan script tradition. ISO 639-5: art."),
    "tok":  ("art",  "Toki Pona",         "Constructed minimalist language. ISO 639-5: art."),
    "zbl":  ("art",  "Blissymbols",       "Graphical symbol-based communication system. ISO 639-5: art."),

    # ── Sign languages ────────────────────────────────────────────────────
    "ase":  ("sgn",  "American Sign Language", "Visual-gestural language of the US and Anglophone Canada. ISO 639-5: sgn."),

    # ── Khoisan ───────────────────────────────────────────────────────────
    "naq":  ("khi",  "Nama",              "Khoe-Kwadi Khoisan language of Namibia/South Africa. ISO 639-5: khi."),

    # ── Miscellaneous / Altaic: Tungusic ──────────────────────────────────
    "evn":  ("tut",  "Evenki",            "Tungusic language of Siberia. ISO 639-5: tut."),
    "mnc":  ("tut",  "Manchu",            "Tungusic language of Manchuria. ISO 639-5: tut."),
    "ude":  ("tut",  "Udihe",             "Tungusic language of the Russian Far East. ISO 639-5: tut."),

    # ── Chukotko-Kamchatkan (no ISO 639-5) ────────────────────────────────
    "ckt":  (None,   "Chukot",            "Chukotko-Kamchatkan language of Chukotka, Russia. No ISO 639-5 code."),
    "kpy":  (None,   "Koryak",            "Chukotko-Kamchatkan language of Kamchatka, Russia. No ISO 639-5 code."),

    # ── Language isolates ─────────────────────────────────────────────────
    "ain":  (None,   "Ainu",              "Language isolate of Hokkaido, Japan. No ISO 639-5 code; Ainu has no established relatives."),
    "hai":  (None,   "Haida",             "Language isolate of Haida Gwaii, Canada. No ISO 639-5 code."),
    "hax":  (None,   "Southern Haida",    "Language isolate of Haida Gwaii. Dialect of Haida; no ISO 639-5 code."),
    "hdn":  (None,   "Northern Haida",    "Language isolate of Haida Gwaii. Dialect of Haida; no ISO 639-5 code."),
    "kut":  (None,   "Kutenai",           "Language isolate of the Kootenai people (BC/Idaho/Montana). No ISO 639-5 code."),
    "sei":  (None,   "Seri",              "Language isolate of coastal Sonora, Mexico. No ISO 639-5 code."),
    "was":  (None,   "Washo",             "Language isolate of the Great Basin. No ISO 639-5 code."),
    "zun":  (None,   "Zuni",              "Language isolate of New Mexico. No ISO 639-5 code."),

    # ── Ancient / extinct isolates ────────────────────────────────────────
    "akz":  (None,   "Alabama",           "Muskogean language of Texas (reassigned to nai)."),
    "elx":  (None,   "Elamite",           "Ancient language of Elam; considered a language isolate or unclassified. No ISO 639-5 code."),
    "ett":  (None,   "Etruscan",          "Language isolate of ancient Italy. No ISO 639-5 code."),
    "lab":  (None,   "Linear A",          "Undeciphered writing system of Minoan Crete. No ISO 639-5 code."),
    "osc":  ("ine",  "Oscan",             "Extinct Italic Indo-European language of ancient Italy. ISO 639-5: ine."),
    "sux":  (None,   "Sumerian",          "Language isolate of ancient Mesopotamia. No ISO 639-5 code."),
    "xmr":  (None,   "Meroitic",          "Language of ancient Nubia; affiliation disputed, treated as isolate. No ISO 639-5 code."),
    "xum":  ("ine",  "Umbrian",           "Extinct Italic Indo-European language of ancient Umbria. ISO 639-5: ine."),

    # ── Additional corrections / stragglers ───────────────────────────────
    "akz":  ("nai",  "Alabama",           "Muskogean language of Texas/Alabama. ISO 639-5: nai."),
    "alt":  ("trk",  "Southern Altai",    "Siberian Turkic language of the Altai Republic. ISO 639-5: trk."),
    "amo":  ("nic",  "Amo",               "Plateau Niger-Congo language of Nigeria. ISO 639-5: nic."),
    "aoz":  ("map",  "Uab Meto",          "Austronesian language of West Timor. ISO 639-5: map."),
    "arp":  ("alg",  "Arapaho",           "Algonquian language of the Great Plains. ISO 639-5: alg."),
    "bas":  ("bnt",  "Basaa",             "Bantu language of Cameroon. ISO 639-5: bnt."),
    "bfq":  ("dra",  "Badaga",            "Dravidian language of Tamil Nadu. ISO 639-5: dra."),
    "bin":  ("nic",  "Bini",              "Edoid Niger-Congo language of Nigeria. ISO 639-5: nic."),
    "bla":  ("alg",  "Siksiká",           "Algonquian (Blackfoot) language of Canada. ISO 639-5: alg."),
    "bqi":  ("ira",  "Bakhtiari",         "Southwest Iranian language of the Bakhtiari people. ISO 639-5: ira."),
    "brh":  ("dra",  "Brahui",            "Northern Dravidian language of Balochistan. ISO 639-5: dra."),
    "btv":  ("ira",  "Bateri",            "Dardic Indo-Aryan language of Pakistan. ISO 639-5: ira (dardic branch)."),
    "buc":  ("bnt",  "Bushi",             "Bantu language of Mayotte. ISO 639-5: bnt."),
    "bze":  ("ssa",  "Jenaama Bozo",      "Nilo-Saharan language of Mali. ISO 639-5: ssa."),
    "bzx":  ("ssa",  "Kɛlɛngaxo Bozo",   "Nilo-Saharan language of Mali. ISO 639-5: ssa."),
    "car":  ("sai",  "Carib",             "Cariban language of South America. ISO 639-5: sai."),
    "cch":  ("nic",  "Atsam",             "Plateau Niger-Congo language of Nigeria. ISO 639-5: nic."),
    "ccp":  ("sit",  "Chakma",            "Tibeto-Burman language of the Chittagong Hill Tracts. ISO 639-5: sit."),
    "chg":  ("trk",  "Chagatai",          "Medieval Chagatai Turkic language. ISO 639-5: trk."),
    "chk":  ("map",  "Chuukese",          "Oceanic Austronesian language of Chuuk, Micronesia. ISO 639-5: map."),
    "chm":  ("fiu",  "Mari",              "Finno-Ugric (Mari) language of Russia. ISO 639-5: fiu."),
    "chp":  ("nai",  "Chipewyan",         "Athabaskan language of the Northwest Territories. ISO 639-5: nai."),
    "cop":  ("afa",  "Coptic",            "Afro-Asiatic language descended from Ancient Egyptian. ISO 639-5: afa."),
    "dar":  ("ccn",  "Dargwa",            "Northeast Caucasian language of Dagestan. ISO 639-5: ccn."),
    "dnj":  ("nic",  "Dan",               "Mande Niger-Congo language of Côte d'Ivoire/Guinea. ISO 639-5: nic."),
    "dtp":  ("map",  "Central Dusun",     "Austronesian language of Sabah, Malaysia. ISO 639-5: map."),
    "dua":  ("bnt",  "Duala",             "Bantu language of Cameroon. ISO 639-5: bnt."),
    "eky":  ("sit",  "Eastern Kayah",     "Tibeto-Burman (Kayah) language of Myanmar. ISO 639-5: sit."),
    "fvr":  ("ssa",  "Fur",               "Nilo-Saharan language of Sudan/Chad. ISO 639-5: ssa."),
    "gay":  ("map",  "Gayo",              "Austronesian language of northern Sumatra. ISO 639-5: map."),
    "gba":  ("nic",  "Gbaya",             "Ubangian Niger-Congo language of Central Africa. ISO 639-5: nic."),
    "gez":  ("sem",  "Geez",              "Semitic liturgical language of Ethiopia/Eritrea. ISO 639-5: sem."),
    "gld":  ("tut",  "Nanai",             "Tungusic language of the Russian Far East. ISO 639-5: tut."),
    "gor":  ("map",  "Gorontalo",         "Austronesian language of Sulawesi. ISO 639-5: map."),
    "gvr":  ("sit",  "Gurung",            "Tibeto-Burman language of Nepal. ISO 639-5: sit."),
    "kac":  ("sit",  "Kachin",            "Tibeto-Burman language of Myanmar/Yunnan. ISO 639-5: sit."),
    "kdt":  ("aav",  "Kuy",               "Austroasiatic (Mon-Khmer) language of Thailand. ISO 639-5: aav."),
    "kea":  ("crp",  "Kabuverdianu",      "Portuguese-based creole of Cape Verde. ISO 639-5: crp."),
    "kge":  ("map",  "Komering",          "Austronesian language of South Sumatra. ISO 639-5: map."),
    "kho":  ("ira",  "Khotanese",         "Eastern Middle Iranian language of ancient Khotan. ISO 639-5: ira."),
    "khq":  ("ssa",  "Koyra Chiini",      "Songhay Nilo-Saharan language of Mali. ISO 639-5: ssa."),
    "kht":  ("sit",  "Khamti",            "Tai-Kadai language (tai) of Myanmar. ISO 639-5: tai."),
    "kmb":  ("bnt",  "Kimbundu",          "Bantu language of Angola. ISO 639-5: bnt."),
    "kos":  ("map",  "Kosraean",          "Oceanic Austronesian language of Kosrae, Micronesia. ISO 639-5: map."),
    "krl":  ("fiu",  "Karelian",          "Finnic Uralic language of Karelia. ISO 639-5: fiu."),
    "kri":  ("crp",  "Krio",              "English-based creole of Sierra Leone. ISO 639-5: crp."),
    "kyu":  ("sit",  "Western Kayah",     "Tibeto-Burman (Kayah) language of Myanmar. ISO 639-5: sit."),
    "lam":  ("bnt",  "Lamba",             "Bantu language of Zambia. ISO 639-5: bnt."),
    "lbw":  ("map",  "Tolaki",            "Austronesian language of Southeast Sulawesi. ISO 639-5: map."),
    "lol":  ("bnt",  "Mongo",             "Bantu language of the DRC. ISO 639-5: bnt."),
    "loz":  ("bnt",  "Lozi",              "Bantu language of Zambia. ISO 639-5: bnt."),
    "ltg":  ("bat",  "Latgalian",         "Baltic Indo-European language of eastern Latvia. ISO 639-5: bat."),
    "mas":  ("ssa",  "Masai",             "Nilotic language of Kenya/Tanzania. ISO 639-5: ssa."),
    "maz":  ("cai",  "Central Mazahua",   "Oto-Manguean language of Mexico. ISO 639-5: cai."),
    "mdh":  ("map",  "Maguindanaon",      "Philippine language of Mindanao. ISO 639-5: map."),
    "mic":  ("alg",  "Mikmaw",            "Algonquian language of the Maritime provinces. ISO 639-5: alg."),
    "mni":  ("sit",  "Manipuri",          "Tibeto-Burman language of Manipur, India. ISO 639-5: sit."),
    "non":  ("gem",  "Old Norse",         "North Germanic language of the Viking Age. ISO 639-5: gem."),
    "nqo":  ("nic",  "NKo",               "Mande language (N'Ko script). ISO 639-5: nic."),
    "nwc":  ("sit",  "Classical Newari",  "Classical form of Newari, a Tibeto-Burman language. ISO 639-5: sit."),
    "nxq":  ("sit",  "Naxi",              "Tibeto-Burman language of Yunnan. ISO 639-5: sit."),
    "pau":  ("map",  "Palauan",           "Malayo-Polynesian language of Palau. ISO 639-5: map."),
    "pko":  ("ssa",  "Pökoot",            "Nilotic language of Kenya/Uganda. ISO 639-5: ssa."),
    "pon":  ("map",  "Pohnpeian",         "Oceanic Austronesian language of Pohnpei, Micronesia. ISO 639-5: map."),
    "rhg":  ("inc",  "Rohingya",          "Indo-Aryan language of the Rohingya people. ISO 639-5: inc."),
    "ria":  ("sit",  "Riang India",       "Tibeto-Burman language of Tripura/Assam. ISO 639-5: sit."),
    "rmf":  ("inc",  "Kalo Finnish Romani","Indo-Aryan (Romani) language spoken in Finland. ISO 639-5: inc."),
    "rmo":  ("inc",  "Sinte Romani",      "Indo-Aryan (Romani) language. ISO 639-5: inc."),
    "rmt":  ("ira",  "Domari",            "Indo-Aryan language of the Dom people (Middle East). ISO 639-5: ira (some debate; inc also used)."),
    "rmu":  ("inc",  "Tavringer Romani",  "Indo-Aryan (Romani) language of Scandinavia. ISO 639-5: inc."),
    "rom":  ("inc",  "Romany",            "Indo-Aryan language of the Romani people. ISO 639-5: inc."),
    "rue":  ("sla",  "Rusyn",             "East Slavic language of the Carpathian region. ISO 639-5: sla."),
    "sad":  (None,   "Sandawe",           "Language isolate (proposed Khoisan link unconfirmed) of Tanzania. No ISO 639-5 code."),
    "saf":  ("nic",  "Safaliba",          "Gur Niger-Congo language of Ghana. ISO 639-5: nic."),
    "saq":  ("ssa",  "Samburu",           "Nilotic language of Kenya. ISO 639-5: ssa."),
    "saz":  ("inc",  "Saurashtra",        "Indo-Aryan language of Gujarat/Tamil Nadu. ISO 639-5: inc."),
    "sel":  ("fiu",  "Selkup",            "Samoyedic Uralic language of western Siberia. ISO 639-5: fiu."),
    "ses":  ("ssa",  "Koyraboro Senni",   "Songhay Nilo-Saharan language of Mali. ISO 639-5: ssa."),
    "sgs":  ("bat",  "Samogitian",        "Baltic Indo-European dialect of Lithuania. ISO 639-5: bat."),
    "srr":  ("nic",  "Serer",             "Atlantic Niger-Congo language of Senegal/Gambia. ISO 639-5: nic."),
    "sus":  ("nic",  "Susu",              "Mande Niger-Congo language of Guinea. ISO 639-5: nic."),
    "tce":  ("nai",  "Southern Tutchone", "Athabaskan language of the Yukon. ISO 639-5: nai."),
    "tmh":  ("ber",  "Tamashek",          "Tuareg Berber language of the Sahara. ISO 639-5: ber."),
    "trv":  ("map",  "Taroko",            "Formosan Austronesian language of Taiwan. ISO 639-5: map."),
    "ukl":  ("sgn",  "Ukrainian Sign Language", "Sign language of Ukraine. ISO 639-5: sgn."),
    "vai":  ("nic",  "Vai",               "Mande Niger-Congo language of Liberia/Sierra Leone. ISO 639-5: nic."),
    "wbp":  ("aus",  "Warlpiri",          "Australian language of the Ngarrka branch, Northern Territory. ISO 639-5: aus."),
    "yao":  ("bnt",  "Yao",               "Bantu language of Malawi/Mozambique/Tanzania. ISO 639-5: bnt."),

    # ── Australian ───────────────────────────────────────────────────────
    "wbp":  ("aus",  "Warlpiri",          "Australian language of the Ngarrka branch. ISO 639-5: aus."),

    # ── Misc remaining ────────────────────────────────────────────────────
    "gag":  ("trk",  "Gagauz",            "Oghuz Turkic language of Moldova/Bulgaria. ISO 639-5: trk."),
    "hoj":  ("inc",  "Hadothi",           "Rajasthani Indo-Aryan dialect of Rajasthan. ISO 639-5: inc."),
    "kaa":  ("trk",  "Kara-Kalpak",       "Kypchak Turkic language of Karakalpakstan. ISO 639-5: trk."),
    "kao":  ("nic",  "Xaasongaxango",     "Mande Niger-Congo language of Senegal/Mali. ISO 639-5: nic."),
    "kaw":  ("map",  "Kawi",              "Old Javanese, an Austronesian literary language of Java. ISO 639-5: map."),
    "kkj":  ("bnt",  "Kako",              "Bantu language of Cameroon. ISO 639-5: bnt."),
    "kln":  ("ssa",  "Kalenjin",          "Nilotic language cluster of Kenya. ISO 639-5: ssa."),
    "dng":  ("sit",  "Dungan",            "Sinitic (Mandarin-based) language of Central Asia. ISO 639-5: sit."),
    "lus":  ("sit",  "Mizo",              "Tibeto-Burman (Kuki-Chin) language of Mizoram, India. ISO 639-5: sit."),
    "lzh":  ("sit",  "Literary Chinese",  "Classical written form of Chinese. ISO 639-5: sit."),
    "noe":  ("inc",  "Nimadi",            "Indo-Aryan language of Madhya Pradesh/Rajasthan. ISO 639-5: inc."),
    "puu":  ("bnt",  "Punu",              "Bantu language of Gabon. ISO 639-5: bnt."),
    "sdc":  ("roa",  "Sassarese Sardinian","Sardinian Romance language of northern Sardinia. ISO 639-5: roa."),
    "ter":  ("sai",  "Tereno",            "Arawakan language of Mato Grosso do Sul, Brazil. ISO 639-5: sai."),
    "tkl":  ("map",  "Tokelau",           "Polynesian language of Tokelau. ISO 639-5: map."),
    "zea":  ("gem",  "Zeelandic",         "Low Franconian Germanic dialect of Zeeland. ISO 639-5: gem."),
    "zmi":  ("map",  "Negeri Sembilan Malay", "Austronesian language of Negeri Sembilan, Malaysia. ISO 639-5: map."),
}

# ---------------------------------------------------------------------------
# iso639_5 -> family_name mapping
# ---------------------------------------------------------------------------
ISO5_TO_FAMILY = {
    # Indo-European (all sub-branches map to the same top-level family)
    "ine": "Indo-European languages",
    "gem": "Indo-European languages",
    "roa": "Indo-European languages",
    "sla": "Indo-European languages",
    "bat": "Indo-European languages",
    "cel": "Indo-European languages",
    "inc": "Indo-European languages",
    "ira": "Indo-European languages",
    "grk": "Indo-European languages",
    "hyx": "Indo-European languages",
    "sqj": "Indo-European languages",
    # Sino-Tibetan
    "sit": "Sino-Tibetan languages",
    # Tai
    "tai": "Tai languages",
    # Japonic
    "jpx": "Japanese languages",
    # Austronesian
    "map": "Austronesian languages",
    # Austro-Asiatic
    "aav": "Austro-Asiatic languages",
    # Hmong-Mien
    "hmx": "Hmong-Mien languages",
    # Altaic / Turkic / Mongolian / Tungusic
    "trk": "Altaic languages",
    "xgn": "Altaic languages",
    "tut": "Altaic languages",
    # Uralic
    "urj": "Uralic languages",
    "fiu": "Uralic languages",
    "smi": "Uralic languages",
    # Dravidian
    "dra": "Dravidian languages",
    # Niger-Kordofanian (all sub-branches)
    "nic": "Niger-Kordofanian languages",
    "bnt": "Niger-Kordofanian languages",
    # Afro-Asiatic (all sub-branches)
    "afa": "Afro-Asiatic languages",
    "sem": "Afro-Asiatic languages",
    "cus": "Afro-Asiatic languages",
    "ber": "Afro-Asiatic languages",
    "cdc": "Afro-Asiatic languages",
    # Nilo-Saharan
    "ssa": "Nilo-Saharan languages",
    # North American Indian
    "nai": "North American Indian languages",
    "alg": "North American Indian languages",
    # Central American Indian
    "cai": "Central American Indian languages",
    # South American Indian
    "sai": "South American Indian languages",
    # Eskimo-Aleut
    "esx": "Eskimo-Aleut languages",
    # Australian
    "aus": "Australian languages",
    # Khoisan
    "khi": "Khoisan languages",
    # Caucasian
    "cau": "Caucasian languages",
    "ccn": "Caucasian languages",
    "ccs": "Caucasian languages",
    # Creoles and pidgins
    "crp": "Creoles and pidgins",
    # Artificial
    "art": "Artificial languages",
    # Papuan
    "paa": "Papuan languages",
    # Sign languages
    "sgn": "Sign languages",
}

# Special family_name for null iso639_5 entries, keyed by language code
NULL_FAMILY_NAMES = {
    "ain":  "Language isolate",
    "hai":  "Language isolate",
    "hax":  "Language isolate",
    "hdn":  "Language isolate",
    "kut":  "Language isolate",
    "sei":  "Language isolate",
    "was":  "Language isolate",
    "zun":  "Language isolate",
    "elx":  "Language isolate",
    "ett":  "Language isolate",
    "sux":  "Language isolate",
    "xmr":  "Language isolate",
    "sad":  "Language isolate",
    "lab":  "Undeciphered script",
    "ckt":  "Chukotko-Kamchatkan languages",
    "kpy":  "Chukotko-Kamchatkan languages",
}


def build_entry(code, lang_name, iso5, rationale, null_family_override=None):
    """Build a single JSON entry dict."""
    if iso5 is not None:
        family = ISO5_TO_FAMILY.get(iso5, f"Unknown (iso639_5={iso5})")
    else:
        family = null_family_override or NULL_FAMILY_NAMES.get(code, "Unclassified")
    return {
        "language_code": code,
        "language_name": lang_name,
        "iso639_5": iso5,
        "family_name": family,
        "rationale": rationale,
    }


def main():
    # ------------------------------------------------------------------
    # 1. Read existing JSON
    # ------------------------------------------------------------------
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)

    existing_codes = {e["language_code"] for e in existing}
    print(f"Existing entries: {len(existing)}")

    # ------------------------------------------------------------------
    # 2. Build new entries from ASSIGNMENTS
    #    (skip codes already in existing JSON)
    # ------------------------------------------------------------------
    new_entries = []
    seen_in_new = set()

    # The canonical target codes from the user's list
    target_codes_str = """abq abr ace ach ada ady aeb afh agq aii ain akk akz ale aln alt
amo ann anp aoz apc arn aro arp arq ars arw ary asa ase atj avk bal
ban bap bas bax bbc bbj bci bej bem bew bez bfd bfq bft bfy bgc bgn
bgx bhb bhi bik bin bjj bjn bkm bku bla blo blt bmq bqi bqv bra brh
bss bto btv bua buc bum bvb byn byv bze bzx cad car cay cch ccp cgg
chb chg chk chm chn chp cic cja cjm cjs ckt clc cnr cop cps crg crh
crj crk crl crm crs csw ctd cwd dak dar dav dcc del den dgr din dje
dng dnj doi dtm dtp dty dua dum dyo dyu dzg ebu efi egl egy eka eky
elx enm esu ett evn ewo fan fat ffm fia fil fit fon frc frm fro frr
frs fud fuq fuv fvr gaa gag gay gba gbz gcr gez gjk gju gld gmh goh
gom gon gor gos grb grc grt gsw gub guc gur guz gvr gwi hai hax haz
hdn hif hil hit hmd hmn hnd hne hnj hnn hno hoc hoj hop hsb hsn hup
hur iba ibb ife ike ikt izh jam jgo jmc jml jpr jrb jut kaa kac kaj
kam kao kaw kbd kbl kca kcg kck kde kdt kea ken kfo kfr kfy kge kgp
kha khb khn kho khq kht kiu kjg kjh kkj kln kmb koi kok kos kpe kpy
krc kri krj krl kru ksb ksf kum kut kvr kvx kwk kxm kxp kxv kyu lab
lag lah laj lam lan lbe lbw lcp lep lez lfn lif lil lis liv ljp lki
lkt lmn lol lou loz lrc lsm ltg lua lui lun luo lus lut luy luz lwl
lzh mad maf mag mai mak mas maz mde mdf mdh mdr mdt men mer mfa mfe
mga mgh mgo mgp mgy mic mls mnc mni mns mnw moe moh mos mrd mrj mro
mtr mua mvy mwk mwr mwv mxc mye myv myx myz naq nch ndc ngl nhe nhw
nia nij niu njo nmg nnh nod noe nog non nov nqo nsk nus nwc nxq nym
nyn nyo nzi ojb ojc ojg ojs ojw oka osa osc ota otk pal pau pcd pcm
pdt peo pfl phn pis pko pnt pon pqm prd prg pro puu quc qug raj rap
rar rcf rej rgn rhg ria rif rjs rkt rmf rmo rmt rmu rng rob rof rom
rtm rue rug rup rwk ryu sad saf sah sam saq sas sat saz sba sbp sck
scs sdc sdh see sef seh sei sel ses sga sgs shi shn shu sid skr slh
sli sly sma smj smn smp sms snk sog sou srb srn srr srx ssy stq str
suk sus sux swb swg swv sxn syc syi syl syr szl tab taj tbw tce tcy
tdd tdg tdh tem teo ter tgx thl thq thr tht tig tiv tkl tkr tkt tli
tly tmh tog tok tru trv trw tsd tsg tsi tsj ttj ttm tts ttt tvl twq
tyv tzm ude uga uli umb unr unx vai vep vic vmf vmw vot vro vun wae
was wbp wbq wbr wls wni wtm xav xcr xlc xld xmn xmr xna xnr xog xpr
xsa xsr xum yao yap yav ybb yrk yrl yua zag zap zbl zdj zea zen zgh
zmi zun zza"""

    target_codes = set(target_codes_str.split())

    for code in sorted(target_codes):
        if code in existing_codes:
            print(f"  [skip] {code} already in existing JSON")
            continue
        if code in seen_in_new:
            continue
        if code not in ASSIGNMENTS:
            print(f"  [WARN] No assignment found for code: {code}")
            continue

        iso5, lang_name, rationale = ASSIGNMENTS[code]
        entry = build_entry(code, lang_name, iso5, rationale)
        new_entries.append(entry)
        seen_in_new.add(code)

    print(f"New entries to add: {len(new_entries)}")

    # ------------------------------------------------------------------
    # 3. Write combined JSON
    # ------------------------------------------------------------------
    combined = existing + new_entries
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Written {len(combined)} total entries to {JSON_PATH}")

    # ------------------------------------------------------------------
    # 4. Patch the CSV
    # ------------------------------------------------------------------
    # Build lookup from combined list
    lookup = {}
    for entry in combined:
        lookup[entry["language_code"]] = {
            "iso639_5_family": entry.get("iso639_5"),
            "family_name":     entry["family_name"],
        }

    # Read CSV
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    patched = 0
    for row in rows:
        code = row.get("language_code", "").strip()
        current_family = row.get("family_name", "").strip()
        if current_family in ("", "None") or current_family == "nan":
            current_family = ""
        if not current_family and code in lookup:
            data = lookup[code]
            row["family_name"] = data["family_name"]
            if not row.get("iso639_5_family") and data["iso639_5_family"]:
                row["iso639_5_family"] = data["iso639_5_family"]
            patched += 1

    # Write CSV back
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Patched {patched} rows in {CSV_PATH}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  Existing entries (kept): {len(existing)}")
    print(f"  New entries added:       {len(new_entries)}")
    print(f"  Total in JSON:           {len(combined)}")
    print(f"  CSV rows patched:        {patched}")


if __name__ == "__main__":
    main()
