"""What a customer may need to provide, have or do for an operation of the help centre. A fixed vocabulary (no "other"):
that is what makes the per-cluster aggregation meaningful. Ids are stored in kb_articles.required_inputs."""

INPUTS: dict[str, str] = {
    "phone": "номер телефона",
    "passport": "паспорт владельца номера",
    "my_mts": "вход в приложение «Мой МТС» или личный кабинет",
    "sms_code": "доступ к SMS на этом номере (код подтверждения)",
    "sim": "SIM-карта или её PUK-код",
    "device": "модель телефона, роутера или ТВ-приставки",
    "address": "адрес подключения или проживания",
    "contract": "номер договора или лицевого счёта",
    "bank_card": "банковская карта для оплаты",
    "region": "регион или населённый пункт",
    "country": "страна поездки",
    "charge": "дата и сумма списания",
    "service_name": "название тарифа, услуги или подписки",
    "office_visit": "визит в салон МТС",
}
