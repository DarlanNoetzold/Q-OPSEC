package tech.noetzold.validation_send_api.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.servers.Server;
import io.swagger.v3.oas.models.tags.Tag;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class SwaggerConfig {

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("Validation Send API")
                        .version("1.0.0")
                        .description("")
                        .contact(new Contact()
                                .name("Noetzold Tech")
                                .email("contato@noetzold.tech")
                                .url("https://noetzold.tech"))
                        .license(new License()
                                .name("Apache 2.0")
                                .url("https://www.apache.org/licenses/LICENSE-2.0.html")))
                .servers(List.of(
                        new Server()
                                .url("http://localhost:8005")
                                .description("Servidor de Desenvolvimento"),
                        new Server()
                                .url("https://api.noetzold.tech")
                                .description("Servidor de Produção")));
    }
}
